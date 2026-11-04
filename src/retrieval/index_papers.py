from src.retrieval.chroma_store import ChromaStore


def index_papers():
    store = ChromaStore()

    total_indexed = store.index_saved_embeddings()

    print("Indexing completed successfully.")
    print("Total indexed chunks:", total_indexed)
    print("Total records in database:", store.count())


if __name__ == "__main__":
    index_papers()
