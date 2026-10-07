from app.config import RAW_DATA_DIR
from app.pdf_loader import load_pdf
from app.vector_store import add_documents


def main():
    pdf_files = list(
        RAW_DATA_DIR.glob("*.pdf")
    )
    print(
        f"Found PDFs: {len(pdf_files)}"
    )
    total_chunks = 0
    for pdf_path in pdf_files:
        print(
            f"Processing: {pdf_path.name}"
        )
        records = load_pdf(pdf_path)
        add_documents(records)
        total_chunks += len(records)
        print(f"Chunks: {len(records)}")

    print(
        f"Finished. Total chunks: "
        f"{total_chunks}"
    )

if __name__ == "__main__":
    main()