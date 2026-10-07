from pathlib import Path

import fitz
import pandas as pd

from app.config import METADATA_FILE

def load_metadata():
    if not METADATA_FILE.exists():
        return {}

    df = pd.read_csv(METADATA_FILE)

    return {
        row["filename"]: {
            "title": str(row["title"]),
            "date": str(row["date"]),
            "document_type": str(row["document_type"])
        }
        for _, row in df.iterrows()
    }

def split_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200
):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def load_pdf(pdf_path: Path):
    metadata_map = load_metadata()

    file_metadata = metadata_map.get(
        pdf_path.name,
        {}
    )
    pdf = fitz.open(pdf_path)
    records = []
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        chunks = split_text(text)
        for chunk_number, chunk in enumerate(chunks):
            record_id = (
                f"{pdf_path.stem}"
                f"_p{page_number}"
                f"_c{chunk_number}"
            )

            metadata = {
                "source": pdf_path.name,
                "page": page_number,
                "title": file_metadata.get(
                    "title",
                    pdf_path.stem
                ),
                "date": file_metadata.get(
                    "date",
                    ""
                ),
                "document_type": file_metadata.get(
                    "document_type",
                    "unknown"
                )
            }
            records.append({
                "id": record_id,
                "text": chunk,
                "metadata": metadata
            })
    return records