import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# Initialize Gemini
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

MODEL = "gemini-2.5-flash"


def chatbot(user_input):

    response = client.models.generate_content(
        model=MODEL,
        contents=user_input
    )

    return response.text or "No response generated."


# Chatbot loop
print("=" * 45)
print("  AI CHATBOT - WITHOUT GUARDRAILS")
print("=" * 45)
print("Type 'exit' to stop.\n")


while True:

    user_input = input("YOU: ").strip()

    if user_input.lower() == "exit":
        print("Chatbot stopped.")
        break

    if not user_input:
        print("Please enter a message.")
        continue

    try:
        answer = chatbot(user_input)
        print("\nBOT:", answer, "\n")

    except Exception as error:
        print("\nError:", error, "\n")