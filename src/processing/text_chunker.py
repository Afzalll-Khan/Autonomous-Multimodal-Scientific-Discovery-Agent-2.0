import json
from pathlib import Path


def load_paper_data(json_path):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    return pages


def create_chunks(text, chunk_size=500, overlap=50):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_paper_chunks(pages, paper_id, chunk_size=500, overlap=50):
    all_chunks = []
    chunk_counter = 1

    for page in pages:
        page_chunks = create_chunks(
            page["text"], chunk_size=chunk_size, overlap=overlap
        )

        for chunk in page_chunks:
            all_chunks.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": chunk_counter,
                    "page_number": page["page_number"],
                    "text": chunk,
                }
            )

            chunk_counter += 1

    return all_chunks


def save_chunks(chunks, paper_id):
    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "data" / "processed" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{paper_id}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print(f"Successfully saved {len(chunks)} chunks.")
    return output_file


def process_all_papers(input_dir):
    input_dir = Path(input_dir)

    all_processed_files = []
    total_chunks = 0

    for json_path in input_dir.glob("*.json"):
        paper_id = json_path.stem

        print(f"\nProcessing paper: {paper_id}")

        pages = load_paper_data(json_path)

        chunks = create_paper_chunks(
            pages, paper_id=paper_id, chunk_size=500, overlap=50
        )

        saved_file = save_chunks(chunks, paper_id)

        all_processed_files.append(saved_file)
        total_chunks += len(chunks)

    print("\nAll papers processed!")
    print("Total papers:", len(all_processed_files))
    print("Total chunks:", total_chunks)

    return all_processed_files


if __name__ == "__main__":
    process_all_papers("data/processed")
