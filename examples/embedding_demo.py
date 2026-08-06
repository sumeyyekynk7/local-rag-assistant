import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def main():
    print("Foundry Local başlatılıyor...")

    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Embedding modeli hazırlanıyor...")

    model = manager.catalog.get_model("qwen3-embedding-0.6b")

    model.download(
        lambda progress: print(
            f"\rModel indiriliyor: %{progress:.2f}",
            end="",
            flush=True,
        )
    )

    print()
    model.load()
    print("Embedding modeli yüklendi.")

    embedding_client = model.get_embedding_client()

    documents = [
        "Python bir programlama dilidir.",
        "Kediler evcil hayvanlardır.",
        "RAG, belgelerden ilgili bilgileri bulup yapay zekâ modeline verir.",
        "SQLite yerel ve hafif bir veritabanıdır.",
    ]

    question = "Belgelerin içinde ilgili bilgiyi hangi yöntem bulur?"

    document_response = embedding_client.generate_embeddings(documents)

    document_embeddings = [
        item.embedding for item in document_response.data
    ]

    question_response = embedding_client.generate_embedding(question)
    question_embedding = question_response.data[0].embedding

    scores = []

    for document, document_embedding in zip(
        documents,
        document_embeddings,
    ):
        score = cosine_similarity(
            question_embedding,
            document_embedding,
        )

        scores.append((document, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    print("\nSoru:")
    print(question)

    print("\nBenzerlik sonuçları:")

    for document, score in scores:
        print(f"{score:.4f} → {document}")

    best_document, best_score = scores[0]

    print("\nEn alakalı cümle:")
    print(best_document)

    print(f"\nBenzerlik puanı: {best_score:.4f}")

    model.unload()
    print("\nModel kapatıldı.")


if __name__ == "__main__":
    main()