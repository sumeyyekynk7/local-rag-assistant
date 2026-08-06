import sqlite3
from pathlib import Path


DATABASE_PATH = Path("data/rag_database.db")


def main():
    if not DATABASE_PATH.exists():
        print(f"Veritabanı bulunamadı: {DATABASE_PATH}")
        return

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # Veritabanındaki tabloları göster
    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
    """)

    tables = cursor.fetchall()

    print("Veritabanındaki tablolar:")
    for table in tables:
        print(f"- {table[0]}")

    # Her tablonun sütunlarını göster
    for table in tables:
        table_name = table[0]

        print(f"\n'{table_name}' tablosunun sütunları:")

        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        for column in columns:
            print(
                f"- sütun adı: {column[1]}, "
                f"veri tipi: {column[2]}"
            )

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        print(f"Toplam kayıt: {row_count}")

    connection.close()


if __name__ == "__main__":
    main()