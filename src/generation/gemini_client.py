import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-3.5-flash-lite"


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    client = genai.Client(api_key=api_key)

    return client


def list_available_models():
    client = get_gemini_client()

    for model in client.models.list():
        print(model.name)


def generate_response(prompt):
    client = get_gemini_client()

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

    return response.text


if __name__ == "__main__":
    prompt = "Explain Retrieval-Augmented Generation in simple terms."

    response = generate_response(prompt)

    print("\nGemini Response:\n")
    print(response)
