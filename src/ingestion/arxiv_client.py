import requests
import xml.etree.ElementTree as ET

ARXIV_API_URL = "https://export.arxiv.org/api/query"

ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def search_papers(query, maxx_results=5):

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": maxx_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)

    response.raise_for_status()

    root = ET.fromstring(response.content)

    papers = []

    for entry in root.findall("atom:entry", ATOM_NAMESPACE):
        paper = {
            "title": entry.findtext("atom:title", default="", namespaces=ATOM_NAMESPACE)
            .strip()
            .replace("\n", " "),
            "abstract": entry.findtext(
                "atom:summary", default="", namespaces=ATOM_NAMESPACE
            )
            .strip()
            .replace("\n", " "),
            "published": entry.findtext(
                "atom:published", default="", namespaces=ATOM_NAMESPACE
            ),
            "authors": [
                author.findtext("atom:name", default="", namespaces=ATOM_NAMESPACE)
                for author in entry.findall("atom:author", ATOM_NAMESPACE)
            ],
            "arxiv_url": entry.findtext(
                "atom:id", default="", namespaces=ATOM_NAMESPACE
            ),
        }
        papers.append(paper)
    return papers


if __name__ == "__main__":
    results = search_papers(query="machine learning", maxx_results=5)

    for index, paper in enumerate(results, start=1):
        print(f"\nPaper {index}")
        print("-" * 60)

        print("Title:", paper["title"])
        print("Authors:", ", ".join(paper["authors"]))
        print("Published:", paper["published"])
        print("arXiv:", paper["arxiv_url"])
        print("Abstract:", paper["abstract"][:300], "...")
