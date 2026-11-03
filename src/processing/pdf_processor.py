import json
import pymupdf
from pathlib import Path


def extract_text(pdf_path):
    extracted_pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")

            extracted_pages.append({"page_number": page_number, "text": page_text})

    return extracted_pages


def save_extracted_text(extracted_pages, pdf_path):
    pdf_path = Path(pdf_path)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{pdf_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(extracted_pages, file, indent=4, ensure_ascii=False)

    print(f"Successfully saved extracted text: {output_file}")

    return output_file


def process_multiple_pdfs(pdf_dir):
    pdf_dir = Path(pdf_dir)

    processed_files = []

    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"\nProcessing: {pdf_path.name}")

        pages = extract_text(pdf_path)

        output_file = save_extracted_text(pages, pdf_path)

        processed_files.append(output_file)

    return processed_files


if __name__ == "__main__":
    processed_files = process_multiple_pdfs("data/raw/pdfs")

    print("\nProcessing completed!")
    print("Total PDFs processed:", len(processed_files))
