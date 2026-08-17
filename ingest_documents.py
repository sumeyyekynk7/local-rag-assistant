import json
import sqlite3
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from console_utils import configure_utf8_output
from document_loader import iter_supported_documents


DOCUMENTS_DIRECTORY = Path("data/documents")
DATABASE_PATH = Path("data/rag_database.db")


def create_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )

        connection.commit()


def clear_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM document_chunks")
        connection.commit()


def split_into_chunks(text: str) -> list[str]:
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        cleaned_paragraph = paragraph.strip()

        if cleaned_paragraph:
            chunks.append(cleaned_paragraph)

    return chunks


def save_chunk(
    source: str,
    chunk_index: int,
    content: str,
    embedding: list[float],
) -> None:
    embedding_json = json.dumps(embedding)

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO document_chunks (
                source,
                chunk_index,
                content,
                embedding
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                source,
                chunk_index,
                content,
                embedding_json,
            ),
        )

        connection.commit()


def main() -> None:
    configure_utf8_output()

    print("Veritabanı hazırlanıyor...")

    create_database()
    clear_database()

    print("Belgeler okunuyor...")

    documents = iter_supported_documents(DOCUMENTS_DIRECTORY)

    if not documents:
        print(
            "data/documents klasöründe desteklenen belge bulunamadı "
            "(.txt, .docx, .pdf)."
        )
        return

    print(f"{len(documents)} belge bulundu.")

    print("Foundry Local başlatılıyor...")

    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    print("Embedding modeli hazırlanıyor...")

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.download()
    model.load()

    embedding_client = model.get_embedding_client()

    total_chunks = 0

    for document in documents:
        source = document["source"]
        chunks = split_into_chunks(document["content"])

        print(f"\n{source}: {len(chunks)} parça bulundu.")

        response = embedding_client.generate_embeddings(chunks)

        for index, embedding_item in enumerate(response.data):
            save_chunk(
                source=source,
                chunk_index=index,
                content=chunks[index],
                embedding=embedding_item.embedding,
            )

            total_chunks += 1

    model.unload()

    print("\nBelge işleme tamamlandı.")
    print(f"Toplam belge sayısı: {len(documents)}")
    print(f"Toplam parça sayısı: {total_chunks}")
    print(f"Veritabanı: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
