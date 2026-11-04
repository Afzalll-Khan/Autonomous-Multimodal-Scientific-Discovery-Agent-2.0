from src.retrieval.chroma_store import ChromaStore
from src.retrieval.embedding_generator import load_embedding_model


def search_papers(query, n_results=5):
    model = load_embedding_model()
    store = ChromaStore()

    results = store.search_text(query=query, model=model, n_results=n_results)

    return results


if __name__ == "__main__":
    query = "How does active learning improve machine learning on continuously arriving data streams?"

    results = search_papers(query, n_results=5)

    for index, result in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Result {index}")
        print(f"{'=' * 60}")

        print("Paper ID:", result["metadata"]["paper_id"])
        print("Chunk ID:", result["metadata"]["chunk_id"])
        print("Page:", result["metadata"]["page_number"])
        print("Similarity:", result.get("similarity"))
        print("\nRetrieved Text:")
        print(result["document"])
