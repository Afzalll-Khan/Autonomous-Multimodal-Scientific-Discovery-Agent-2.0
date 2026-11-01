from src.ingestion.arxiv_client import search_papers
import requests
from pathlib import Path


def get_pdf_url(arxiv_url):
    paper_id = arxiv_url.rstrip("/").split("/")[-1]
    return f"https://arxiv.org/pdf/{paper_id}"


def download_pdf(arxiv_url, output_dir):
    pdf_url = get_pdf_url(arxiv_url)

    paper_id = arxiv_url.rstrip("/").split("/")[-1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{paper_id}.pdf"

    headers = {"User-Agent": "ScientificDiscoveryAgent/1.0"}

    if output_file.exists():
        print(f"Already exists: {paper_id}.pdf")
        return output_file

    response = requests.get(pdf_url, headers=headers, timeout=60)
    response.raise_for_status()

    with open(output_file, "wb") as file:
        file.write(response.content)

    print(f"Downloaded: {paper_id}.pdf")

    return output_file


def download_multiple_papers(papers, output_dir):
    downloaded_files = []

    for paper in papers:
        arxiv_url = paper["arxiv_url"]

        pdf_path = download_pdf(arxiv_url, output_dir)

        downloaded_files.append(pdf_path)

    return downloaded_files


if __name__ == "__main__":
    papers = search_papers(query="machine learning", max_results=5)

    downloaded_files = download_multiple_papers(
        papers=papers, output_dir="data/raw/pdfs"
    )

    print(f"\nTotal papers processed: {len(downloaded_files)}")
