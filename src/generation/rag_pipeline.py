import json
from pathlib import Path

from src.retrieval.search_papers import search_papers
from src.generation.gemini_client import generate_response


# --------------------------------------------------
# 1. Load paper metadata
# --------------------------------------------------


def load_paper_metadata():
    project_root = Path(__file__).resolve().parents[2]
    papers_path = project_root / "data" / "raw" / "papers.json"

    if not papers_path.exists():
        raise FileNotFoundError(f"Paper metadata file not found: {papers_path}")

    with open(papers_path, "r", encoding="utf-8") as file:
        papers = json.load(file)

    paper_metadata = {}

    for paper in papers:
        arxiv_url = paper.get("arxiv_url")

        if not arxiv_url:
            continue

        paper_id = arxiv_url.rstrip("/").split("/")[-1]
        paper_metadata[paper_id] = paper

    return paper_metadata


# --------------------------------------------------
# 2. Build research context
# --------------------------------------------------


def build_context(results, paper_metadata):
    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        paper_id = metadata["paper_id"]

        paper = paper_metadata.get(paper_id, {})

        authors = paper.get("authors", [])
        authors_text = ", ".join(authors) if authors else "Unknown"

        source = f"""
Source {index}:

Title: {paper.get("title", "Unknown")}
Authors: {authors_text}
Paper ID: {paper_id}
Page: {metadata.get("page_number", "Unknown")}
Published: {paper.get("published", "Unknown")}
arXiv URL: {paper.get("arxiv_url", "Unknown")}
Similarity: {result.get("similarity", 0):.4f}

Content:
{result["document"]}
"""

        context_parts.append(source)

    return "\n".join(context_parts)


# --------------------------------------------------
# 3. Generate research answer using RAG
# --------------------------------------------------


def generate_research_answer(question, n_results=5):

    # Step 1: Retrieve relevant research passages
    results = search_papers(query=question, n_results=n_results)

    if not results:
        return {
            "question": question,
            "answer": "No relevant research passages were found.",
            "sources": [],
        }

    # Step 2: Load paper metadata
    paper_metadata = load_paper_metadata()

    # Step 3: Build research context
    context = build_context(results=results, paper_metadata=paper_metadata)

    # Step 4: Construct the prompt
    prompt = f"""
You are a scientific research assistant.

Answer the research question using ONLY the provided research context.

If the context does not contain enough information to answer
the question, clearly state that the available literature
does not provide sufficient information.

Do not invent scientific findings, paper details, or references.

Research Context:
{context}

Research Question:
{question}

Instructions:
1. Provide a clear and scientifically accurate answer.
2. Organize the answer logically.
3. Reference the source numbers when making claims.
4. Distinguish between established findings and limitations.
5. Do not use information outside the provided research context.

Answer:
"""

    # Step 5: Generate answer using Gemini
    answer = generate_response(prompt)

    return {"question": question, "answer": answer, "sources": results}


# --------------------------------------------------
# 4. Run the RAG pipeline
# --------------------------------------------------

if __name__ == "__main__":
    question = (
        "How does active learning improve machine learning "
        "on continuously arriving data streams?"
    )

    response = generate_research_answer(question)

    print("\n" + "=" * 70)
    print("RESEARCH QUESTION")
    print("=" * 70)
    print(response["question"])

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(response["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    paper_metadata = load_paper_metadata()

    for index, source in enumerate(response["sources"], start=1):
        metadata = source["metadata"]
        paper_id = metadata["paper_id"]

        paper = paper_metadata.get(paper_id, {})

        print(f"\nSource {index}")
        print("-" * 50)

        print("Title:", paper.get("title", "Unknown"))
        print("Authors:", ", ".join(paper.get("authors", [])))
        print("Paper ID:", paper_id)
        print("Page:", metadata.get("page_number", "Unknown"))
        print("Published:", paper.get("published", "Unknown"))
        print("arXiv URL:", paper.get("arxiv_url", "Unknown"))
        print(f"Similarity: {source.get('similarity', 0):.4f}")
