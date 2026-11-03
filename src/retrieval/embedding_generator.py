import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    model = SentenceTransformer(MODEL_NAME)
    return model


def generate_embedding(model, text):
    embedding = model.encode(text)
    return embedding


def generate_embeddings(model, texts):
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    return embeddings


def load_all_chunks(input_dir):
    input_dir = Path(input_dir)
    all_chunks = []

    for json_path in input_dir.glob("*_chunks.json"):
        with open(json_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)
            all_chunks.extend(chunks)

    return all_chunks


def embed_all_chunks(model, chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({**chunk, "embedding": embedding})

    return embedded_chunks


def save_embeddings(embedded_chunks):
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "vector_store"
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks])

    metadata = [
        {key: value for key, value in chunk.items() if key != "embedding"}
        for chunk in embedded_chunks
    ]

    np.save(output_dir / "embeddings.npy", embeddings)

    with open(output_dir / "embedding_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\nEmbeddings and metadata saved successfully.")
    print("Saved embedding shape:", embeddings.shape)
    print("Saved metadata records:", len(metadata))


def verify_saved_embeddings():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "vector_store"

    saved_embeddings = np.load(output_dir / "embeddings.npy")

    with open(output_dir / "embedding_metadata.json", "r", encoding="utf-8") as file:
        saved_metadata = json.load(file)
    assert saved_embeddings.shape[0] == len(saved_metadata)
    assert saved_embeddings.shape[1] == 384
    print("All embedding integrity checks passed.")
    print("\nVerification:")
    print("Loaded embedding shape:", saved_embeddings.shape)
    print("Loaded metadata records:", len(saved_metadata))
    print("First paper ID:", saved_metadata[0]["paper_id"])
    print("First chunk ID:", saved_metadata[0]["chunk_id"])


if __name__ == "__main__":
    chunks = load_all_chunks("data/processed/chunks")

    print("Total chunks loaded:", len(chunks))

    model = load_embedding_model()
    embedded_chunks = embed_all_chunks(model, chunks)

    print("\nTotal embedded chunks:", len(embedded_chunks))
    print("Embedding dimensions:", embedded_chunks[0]["embedding"].shape)

    save_embeddings(embedded_chunks)
    verify_saved_embeddings()
