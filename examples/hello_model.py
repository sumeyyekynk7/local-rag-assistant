import openai
from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    print("Foundry Local başlatılıyor...")

    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Gerekli çalışma bileşenleri kontrol ediliyor...")

    current_ep = ""

    def show_progress(ep_name: str, percent: float):
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

    print("Model hazırlanıyor...")

    model = manager.catalog.get_model("qwen2.5-0.5b")

    model.download(
        lambda progress: print(
            f"\rModel indiriliyor: %{progress:.2f}",
            end="",
            flush=True,
        )
    )

    print()
    model.load()

    print("Model yüklendi.")

    manager.start_web_service()
    base_url = f"{manager.urls[0]}/v1"

    client = openai.OpenAI(
        base_url=base_url,
        api_key="none",
    )

    print("\nModelin cevabı:\n")

    response = client.chat.completions.create(
        model=model.id,
        messages=[
            {
                "role": "system",
                "content": "Kısa ve anlaşılır cevap veren yardımcı bir asistansın.",
            },
            {
                "role": "user",
                "content": "RAG nedir? İki cümleyle Türkçe açıkla.",
            },
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(
                chunk.choices[0].delta.content,
                end="",
                flush=True,
            )

    print()

    model.unload()
    manager.stop_web_service()


if __name__ == "__main__":
    main()