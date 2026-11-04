"""End-to-end smoke test for PDF processing through persistent semantic search.

Run from the repository root with:
    python tests/smoke_test.py

The test uses generated sample PDFs and does not call arXiv. The first run may
download the sentence-transformers model configured by the project.
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.pdf_processor import extract_text
from src.processing.text_chunker import create_paper_chunks
from src.retrieval.chroma_store import ChromaStore
from src.retrieval.embedding_generator import embed_all_chunks, load_embedding_model


SAMPLE_PAPERS = {
    "quantum-error-correction": (
        "Quantum error correction protects logical qubits from noise by encoding "
        "quantum information across multiple physical qubits. Surface codes detect "
        "errors through stabilizer measurements and repeated correction cycles."
    ),
    "exoplanet-spectroscopy": (
        "Exoplanet atmospheric spectroscopy measures absorption features in starlight "
        "that passes through a planet's atmosphere. Spectral lines help identify "
        "molecules, temperature, clouds, and atmospheric composition."
    ),
}


def create_sample_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 560, 760), text, fontsize=12)
    document.save(path)
    document.close()


def run_smoke_test() -> None:
    print("1/5 Creating sample scientific PDFs...")
    with TemporaryDirectory(
        prefix=".scientific-discovery-smoke-", dir=PROJECT_ROOT
    ) as temp_dir:
        temp_path = Path(temp_dir)
        chunks = []
        for paper_id, text in SAMPLE_PAPERS.items():
            pdf_path = temp_path / f"{paper_id}.pdf"
            create_sample_pdf(pdf_path, text)
            pages = extract_text(pdf_path)
            assert pages and text.split()[0] in pages[0]["text"], (
                f"PDF text extraction failed for {paper_id}"
            )
            chunks.extend(
                create_paper_chunks(
                    pages,
                    paper_id=paper_id,
                    chunk_size=100,
                    overlap=10,
                )
            )
        assert len(chunks) == len(SAMPLE_PAPERS), "Unexpected chunk count"

        print("2/5 Loading the embedding model and embedding chunks...")
        model = load_embedding_model()
        embedded_chunks = embed_all_chunks(model, chunks)
        assert len(embedded_chunks) == len(chunks)
        assert len(embedded_chunks[0]["embedding"]) == 384

        print("3/5 Writing chunks to persistent ChromaDB...")
        store_path = temp_path / "chroma"
        store = ChromaStore(store_path)
        try:
            inserted = store.add_embedded_chunks(embedded_chunks)
            assert inserted == len(chunks)
            assert store.count() == len(chunks)

            print("4/5 Checking scientific semantic search...")
            queries = {
                "How do surface codes detect and correct errors in logical qubits?":
                    "quantum-error-correction",
                "What can spectral lines tell us about exoplanet atmospheres?":
                    "exoplanet-spectroscopy",
            }
            for query, expected_paper_id in queries.items():
                results = store.search_text(query, model, n_results=1)
                assert results, f"No results returned for query: {query}"
                actual_paper_id = results[0]["metadata"]["paper_id"]
                assert actual_paper_id == expected_paper_id, (
                    f"Query '{query}' returned {actual_paper_id}; "
                    f"expected {expected_paper_id}"
                )
                print(f"   PASS: {query} -> {actual_paper_id}")
        finally:
            store.client.close()

        print("5/5 Reopening ChromaDB and checking persistence...")
        reopened_store = ChromaStore(store_path)
        try:
            assert reopened_store.count() == len(chunks)
            assert reopened_store.search_text(
                "How do surface codes detect and correct errors in logical qubits?",
                model,
                n_results=1,
            )
        finally:
            reopened_store.client.close()

    print("SMOKE TEST PASSED: PDF processing, chunking, embeddings, retrieval, and persistence are working.")


if __name__ == "__main__":
    run_smoke_test()
