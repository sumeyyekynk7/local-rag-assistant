import json
import sqlite3
from pathlib import Path

import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager


DATABASE_PATH = Path("data/rag_database.db")


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a, dtype=float)
    vector_b = np.array(vector_b, dtype=float)

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def create_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def clear_database():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("DELETE FROM document_chunks")

    connection.commit()
    connection.close()


def save_document(source, content, embedding):
    embedding_json = json.dumps(embedding)

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO document_chunks (source, content, embedding)
        VALUES (?, ?, ?)
        """,
        (source, content, embedding_json),
    )

    connection.commit()
    connection.close()


def get_all_documents():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, source, content, embedding
        FROM document_chunks
        """
    )

    rows = cursor.fetchall()
    connection.close()

    documents = []

    for row in rows:
        document = {
            "id": row[0],
            "source": row[1],
            "content": row[2],
            "embedding": json.loads(row[3]),
        }

        documents.append(document)

    return documents


def main():
    print("Veritabanı hazırlanıyor...")

    create_database()
    clear_database()

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
    embedding_client = model.get_embedding_client()

    documents = [
        {
            "source": "python_notes.txt",
            "content": "Python bir programlama dilidir.",
        },
        {
            "source": "animal_notes.txt",
            "content": "Kediler evcil hayvanlardır.",
        },
        {
            "source": "rag_notes.txt",
            "content": (
                "RAG, belgelerden ilgili bilgileri bulup "
                "yapay zekâ modeline verir."
            ),
        },
        {
            "source": "database_notes.txt",
            "content": "SQLite yerel ve hafif bir veritabanıdır.",
        },
    ]

    document_texts = [
        document["content"]
        for document in documents
    ]

    print("Belge embedding'leri oluşturuluyor...")

    document_response = embedding_client.generate_embeddings(
        document_texts
    )

    for document, embedding_item in zip(
        documents,
        document_response.data,
    ):
        save_document(
            source=document["source"],
            content=document["content"],
            embedding=embedding_item.embedding,
        )

    print(f"{len(documents)} kayıt veritabanına eklendi.")

    question = "Belgelerin içinde ilgili bilgiyi hangi yöntem bulur?"

    print("\nSoru:")
    print(question)

    question_response = embedding_client.generate_embedding(question)
    question_embedding = question_response.data[0].embedding

    stored_documents = get_all_documents()

    results = []

    for document in stored_documents:
        score = cosine_similarity(
            question_embedding,
            document["embedding"],
        )

        results.append(
            {
                "source": document["source"],
                "content": document["content"],
                "score": score,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print("\nVeritabanından alınan benzerlik sonuçları:")

    for result in results:
        print(
            f'{result["score"]:.4f} '
            f'→ {result["content"]} '
            f'[{result["source"]}]'
        )

    best_result = results[0]

    print("\nEn alakalı kayıt:")
    print(best_result["content"])

    print("\nKaynak:")
    print(best_result["source"])

    print(f"\nBenzerlik puanı: {best_result['score']:.4f}")

    model.unload()

    print("\nEmbedding modeli kapatıldı.")
    print(f"Veritabanı konumu: {DATABASE_PATH}")


if __name__ == "__main__":
    main()