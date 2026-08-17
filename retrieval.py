import json
import math
import sqlite3
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from console_utils import configure_utf8_output


DATABASE_PATH = Path("data/rag_database.db")
TOP_K = 2


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    İki embedding vektörü arasındaki cosine similarity değerini hesaplar.

    Sonuç 1'e ne kadar yakınsa metinler birbirine o kadar benzerdir.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding vektörlerinin boyutları eşit değil.")

    dot_product = sum(
        value_a * value_b
        for value_a, value_b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def load_document_chunks() -> list[dict]:
    """
    SQLite veritabanındaki bütün document chunk kayıtlarını okur.
    """

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Veritabanı bulunamadı: {DATABASE_PATH}"
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                source,
                chunk_index,
                content,
                embedding
            FROM document_chunks
            """
        )

        rows = cursor.fetchall()

    chunks = []

    for row in rows:
        chunks.append(
            {
                "id": row[0],
                "source": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "embedding": json.loads(row[4]),
            }
        )

    return chunks


def get_top_chunks(
    query: str,
    embedding_client,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Kullanıcı sorusuna en yakın document chunk'larını bulur.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        return []

    # Kullanıcı sorusunu embedding vektörüne dönüştür.
    response = embedding_client.generate_embeddings([cleaned_query])
    query_embedding = response.data[0].embedding

    document_chunks = load_document_chunks()

    results = []

    for chunk in document_chunks:
        similarity = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        results.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "similarity": similarity,
            }
        )

    # En yüksek benzerlik değerinden en düşüğe doğru sırala.
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:top_k]


def main() -> None:
    configure_utf8_output()

    print("Foundry Local başlatılıyor...")

    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    print("Embedding modeli hazırlanıyor...")

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.download()
    model.load()

    try:
        embedding_client = model.get_embedding_client()

        print("\nLocal RAG belge arama sistemi hazır.")
        print("Çıkmak için 'q' yazabilirsin.\n")

        while True:
            query = input("Sorun: ").strip()

            if query.lower() == "q":
                print("Program kapatılıyor.")
                break

            if not query:
                print("Lütfen boş olmayan bir soru gir.\n")
                continue

            top_chunks = get_top_chunks(
                query=query,
                embedding_client=embedding_client,
                top_k=TOP_K,
            )

            if not top_chunks:
                print("İlgili belge parçası bulunamadı.\n")
                continue

            print("\nEn ilgili belge parçaları:")

            for rank, result in enumerate(top_chunks, start=1):
                print("\n" + "=" * 60)
                print(f"Sıra: {rank}")
                print(f"Kaynak: {result['source']}")
                print(f"Chunk numarası: {result['chunk_index']}")
                print(f"Benzerlik: {result['similarity']:.4f}")
                print(f"İçerik:\n{result['content']}")

            print("\n" + "=" * 60 + "\n")

    finally:
        model.unload()


if __name__ == "__main__":
    main()
