from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

load_dotenv()


# --------------------------------
# 1. Load Knowledge Base
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

loader = TextLoader(BASE_DIR / "knowledge.txt")
documents = loader.load()

print("Document loaded successfully!")


# --------------------------------
# 2. Split Document into Chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# --------------------------------
# 3. Create Embeddings
# --------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

print("Embedding model initialized!")


# --------------------------------
# 4. Store in ChromaDB
# --------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_knowledge_base"
)

print("Knowledge base stored in ChromaDB!")


# --------------------------------
# 5. Initialize Gemini
# --------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=200
)


# --------------------------------
# 6. RAG Chatbot
# --------------------------------

while True:

    query = input("\nYOU: ")

    if query == "0":
        break

    # Retrieve relevant documents
    results = vectorstore.similarity_search(
        query,
        k=3
    )

    # Combine retrieved information
    context = "\n\n".join(
        result.page_content
        for result in results
    )

    # Ask Gemini using retrieved context
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the information
provided in the knowledge base below.

If the answer is not available in the knowledge base,
say exactly:

"I don't have relevant information about this in my knowledge base."

Knowledge Base:
{context}

User Question:
{query}
"""

    response = model.invoke(prompt)

    print("\nBOT:", response.content)