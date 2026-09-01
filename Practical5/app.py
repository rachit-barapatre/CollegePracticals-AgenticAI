from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


# ---------------------------------------
# 1. Load Environment Variables
# ---------------------------------------

load_dotenv()


# ---------------------------------------
# 2. Load Knowledge Base
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent

loader = TextLoader(
    BASE_DIR / "knowledge_base.txt"
)

documents = loader.load()

print("Knowledge base loaded successfully!")


# ---------------------------------------
# 3. Split Document into Chunks
# ---------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ---------------------------------------
# 4. Create Embeddings
# ---------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

print("Embedding model initialized!")


# ---------------------------------------
# 5. Create ChromaDB
# ---------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="practical5_knowledge_base"
)

print("Knowledge base stored in ChromaDB!")


# ---------------------------------------
# 6. Initialize Gemini
# ---------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=300
)


# ---------------------------------------
# 7. Question Answering Agent
# ---------------------------------------

while True:

    question = input("\nYOU: ")

    if question == "0":
        print("Exiting...")
        break

    # Retrieve relevant information
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    # Combine retrieved documents
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # Create prompt
    prompt = f"""
You are a Document Question Answering Agent.

Your task is to answer the user's question using ONLY
the information available in the provided knowledge base.

Do not use outside knowledge.

If the answer is not available in the knowledge base,
respond with:

"I don't have enough information in the knowledge base
to answer this question."

Knowledge Base:
{context}

User Question:
{question}

Answer:
"""

    # Generate answer
    response = model.invoke(prompt)

    print("\nBOT:", response.content)