import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf"}
WORD_NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8").strip()


def read_docx(file_path: Path) -> str:
    with zipfile.ZipFile(file_path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ElementTree.fromstring(document_xml)
    paragraphs = []

    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        text_parts = [
            node.text
            for node in paragraph.findall(".//w:t", WORD_NAMESPACE)
            if node.text
        ]

        text = "".join(text_parts).strip()

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs).strip()


def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(text)

    return "\n\n".join(pages).strip()


def read_document(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return read_txt(file_path)

    if suffix == ".docx":
        return read_docx(file_path)

    if suffix == ".pdf":
        return read_pdf(file_path)

    raise ValueError(f"Desteklenmeyen belge türü: {file_path.suffix}")


def iter_supported_documents(directory: Path) -> list[dict]:
    documents = []

    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = read_document(file_path)

        if not content:
            continue

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents
