"""
Semantic Chunking + LangChain + ChromaDB Demo
Author: Suhas Training Demo

Goal:
1. Load a text file.
2. Split it using semantic chunking.
3. Store chunks in ChromaDB.
4. Retrieve relevant chunks for user queries.

Run:
    pip install -r requirements_semantic_chromadb.txt
    python semantic_chunking_chromadb_demo.py

Optional:
    For OpenAI embeddings, set:
        export EMBEDDING_PROVIDER=openai
        export OPENAI_API_KEY="your-key"

Default:
    Uses HuggingFace sentence-transformer embeddings locally.
"""

import os
from pathlib import Path

# -----------------------------
# 1. Configuration
# -----------------------------
TEXT_FILE = "semantic_chunking_experiment_text.txt"
PERSIST_DIR = "chroma_semantic_db"
COLLECTION_NAME = "semantic_chunking_demo"

# SemanticChunker threshold options:
# - "percentile" : split where semantic distance is above percentile
# - "standard_deviation" : split where distance exceeds std deviation threshold
# - "interquartile" : split using IQR
BREAKPOINT_TYPE = "percentile"
BREAKPOINT_AMOUNT = 80


# -----------------------------
# 2. Select Embedding Model
# -----------------------------
def get_embeddings():
    """Return embedding model. Uses local HuggingFace by default, OpenAI if configured."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower().strip()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Local embedding model for classroom/lab experiments
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -----------------------------
# 3. Load Text Document
# -----------------------------
def load_document(file_path: str):
    from langchain_community.document_loaders import TextLoader

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from {file_path}")
    return docs


# -----------------------------
# 4. Semantic Chunking
# -----------------------------
def create_semantic_chunks(docs, embeddings):
    from langchain_experimental.text_splitter import SemanticChunker

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=BREAKPOINT_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_AMOUNT,
    )

    chunks = semantic_splitter.split_documents(docs)

    print("\n================ SEMANTIC CHUNKS CREATED ================")
    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.page_content.replace("\n", " ")[:250]
        print(f"\n--- Chunk {i} ---")
        print(preview + "...")

    return chunks


# -----------------------------
# 5. Store Chunks in ChromaDB
# -----------------------------
def store_in_chromadb(chunks, embeddings):
    from langchain_chroma import Chroma

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print("\n================ CHROMADB STORAGE COMPLETE ================")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Persist directory: {PERSIST_DIR}")
    return vectorstore


# -----------------------------
# 6. Similarity Search
# -----------------------------
def retrieve_context(vectorstore, query: str, k: int = 3):
    print("\n================ QUERY ================")
    print(query)

    results = vectorstore.similarity_search_with_score(query, k=k)

    print("\n================ TOP RETRIEVED CHUNKS ================")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\nResult {rank} | Similarity distance: {score:.4f}")
        print(doc.page_content[:800])

    return [doc for doc, score in results]


# -----------------------------
# 7. Optional RAG Answer
# -----------------------------
def generate_answer_with_llm(query: str, retrieved_docs):
    """
    Optional: Uses OpenAI chat model only if OPENAI_API_KEY is available.
    If not available, this function returns retrieved context only.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not found. Skipping final LLM answer.")
        return None

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI trainer. Answer the question using only the provided context.
If the answer is not present, say that the context does not contain enough information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    print("\n================ FINAL RAG ANSWER ================")
    print(response.content)
    return response.content


# -----------------------------
# 8. Main Experiment
# -----------------------------
def main():
    embeddings = get_embeddings()
    docs = load_document(TEXT_FILE)
    chunks = create_semantic_chunks(docs, embeddings)
    vectorstore = store_in_chromadb(chunks, embeddings)

    test_queries = [
        "What is semantic chunking and why is it useful?",
        "How does ChromaDB support RAG?",
        "How can agentic AI use a knowledge base?",
        "What security controls are important in enterprise RAG?",
    ]

    for query in test_queries:
        retrieved_docs = retrieve_context(vectorstore, query, k=3)
        generate_answer_with_llm(query, retrieved_docs)


if __name__ == "__main__":
    main()
