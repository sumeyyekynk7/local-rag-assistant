import openai

from foundry_local_sdk import Configuration, FoundryLocalManager

from retrieval import get_top_chunks


EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
CHAT_MODEL_NAME = "qwen3.5-2b-text"
TOP_K = 2
MIN_SIMILARITY = 0.45


def create_context(retrieved_chunks: list[dict]) -> str:
    """
    Bulunan belge parçalarını LLM'e verilecek bağlam metnine dönüştürür.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
Kaynak {index}
Dosya: {chunk["source"]}
Parça numarası: {chunk["chunk_index"]}
İçerik:
{chunk["content"]}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def print_sources(retrieved_chunks: list[dict]) -> None:
    """
    Cevap oluşturulurken kullanılan kaynakları terminalde gösterir.
    """

    print("\nKullanılan kaynaklar:")

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(
            f"{index}. {chunk['source']} "
            f"(chunk {chunk['chunk_index']}, "
            f"benzerlik: {chunk['similarity']:.4f})"
        )


def context_can_answer(
    question: str,
    context: str,
    client: openai.OpenAI,
    chat_model_id: str,
) -> bool:
    system_prompt = """
Sen bir belge uygunluk kontrolcüsüsün.

Görevin, verilen belge bağlamının kullanıcı sorusunu doğrudan cevaplayıp
cevaplamadığını belirlemektir.

Kurallar:
- Cevap bağlamda açıkça bulunuyorsa yalnızca EVET yaz.
- Cevap bağlamda bulunmuyorsa yalnızca HAYIR yaz.
- Genel bilgini kullanma.
- Benzer kelimeler bulunması yeterli değildir.
- Soruda geçen kişi, tarih, isim veya özel bilgi bağlamda açıkça yoksa HAYIR yaz.
""".strip()

    user_prompt = f"""
BELGE BAĞLAMI:

{context}

SORU:

{question}
""".strip()

    response = client.chat.completions.create(
        model=chat_model_id,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        max_tokens=10,
    )

    decision = response.choices[0].message.content.strip().upper()

    return decision.startswith("EVET")


def generate_answer(
    question: str,
    context: str,
    client: openai.OpenAI,
    chat_model_id: str,
) -> None:
    """
    Kullanıcı sorusu ve bulunan belge bağlamını yerel LLM'e gönderir.
    """

    system_prompt = """
Sen yalnızca verilen belge parçalarındaki bilgileri kullanan bir belge soru-cevap sistemisin.

Kesin kurallar:
1. Cevabını yalnızca BELGE BAĞLAMI içindeki bilgilerden oluştur.
2. Belgede yazan terimleri, isimleri ve açılımları aynen kullan.
3. Bağlamda olmayan hiçbir bilgi ekleme.
4. Kendi genel bilgini kesinlikle kullanma.
5. Belge bağlamı soruyla yalnızca dolaylı olarak ilişkiliyse cevap üretme.
6. Sorunun cevabı bağlamda yoksa yalnızca şu cümleyi yaz:
   Bu sorunun cevabı mevcut belgelerde bulunmuyor.
7. Cevabı Türkçe, kısa ve doğrudan yaz.
8. Talimatları veya kuralları cevabın içinde tekrar etme.
""".strip()

    user_prompt = f"""
BELGE BAĞLAMI:

{context}

SORU:

{question}

Soruyu yalnızca yukarıdaki belge bağlamına dayanarak cevapla.
""".strip()

    response = client.chat.completions.create(
        model=chat_model_id,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        max_tokens=200,
        stream=True,
    )

    for chunk in response:
        if (
            chunk.choices
            and chunk.choices[0].delta.content is not None
        ):
            print(
                chunk.choices[0].delta.content,
                end="",
                flush=True,
            )

    print()


def main() -> None:
    print("Foundry Local başlatılıyor...")

    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    print("Gerekli çalışma bileşenleri kontrol ediliyor...")

    current_ep = ""

    def show_progress(ep_name: str, percent: float) -> None:
        nonlocal current_ep

        if ep_name != current_ep:
            if current_ep:
                print()

            current_ep = ep_name

        print(
            f"\r{ep_name:<30} %{percent:5.1f}",
            end="",
            flush=True,
        )

    manager.download_and_register_eps(
        progress_callback=show_progress
    )

    if current_ep:
        print()

    print("Embedding modeli hazırlanıyor...")

    embedding_model = manager.catalog.get_model(
        EMBEDDING_MODEL_NAME
    )
    embedding_model.download()
    embedding_model.load()

    print("Sohbet modeli hazırlanıyor...")

    chat_model = manager.catalog.get_model(
        CHAT_MODEL_NAME
    )

    chat_model.download(
        lambda progress: print(
            f"\rSohbet modeli indiriliyor: %{progress:.2f}",
            end="",
            flush=True,
        )
    )

    print()
    chat_model.load()

    try:
        embedding_client = (
            embedding_model.get_embedding_client()
        )

        manager.start_web_service()

        base_url = f"{manager.urls[0]}/v1"

        chat_client = openai.OpenAI(
            base_url=base_url,
            api_key="none",
        )

        print("\nLocal RAG Assistant hazır.")
        print("Çıkmak için 'q' yazabilirsin.\n")

        while True:
            question = input("Sorun: ").strip()

            if question.lower() == "q":
                print("Program kapatılıyor.")
                break

            if not question:
                print("Lütfen boş olmayan bir soru gir.\n")
                continue

            print("\nİlgili belge parçaları aranıyor...")

            retrieved_chunks = get_top_chunks(
                query=question,
                embedding_client=embedding_client,
                top_k=TOP_K,
            )

            if not retrieved_chunks:
                print(
                    "\nBu sorunun cevabı mevcut belgelerde bulunmuyor.\n"
                )
                continue

            best_similarity = retrieved_chunks[0]["similarity"]
            print(f"En yüksek benzerlik puanı: {best_similarity:.4f}")

            if best_similarity < MIN_SIMILARITY:
                print(
                    "\nBu sorunun cevabı mevcut belgelerde bulunmuyor.\n"
                )
                continue

            context = create_context(retrieved_chunks)

            print("\nAsistanın cevabı:\n")

            generate_answer(
                question=question,
                context=context,
                client=chat_client,
                chat_model_id=chat_model.id,
            )

            print_sources(retrieved_chunks)

            print("\n" + "=" * 60 + "\n")

    finally:
        print("\nModeller kapatılıyor...")

        embedding_model.unload()
        chat_model.unload()
        manager.stop_web_service()


if __name__ == "__main__":
    main()
