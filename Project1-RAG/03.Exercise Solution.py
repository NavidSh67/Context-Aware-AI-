"""
Exercise: SOLUTION

This is the complete solution with semantic search.
Try to complete exercise.py yourself first before looking here!

This follows the EXACT same pattern as RAG-Project.py.
"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_documents(knowledge_dir: str = "bookstore_knowledge") -> dict:
    """
    Load all .md files from the knowledge directory.
    """
    print(f"\nLoading documents from {knowledge_dir}/...")
    
    knowledge_base = {}
    knowledge_path = Path(__file__).parent / knowledge_dir
    
    if not knowledge_path.exists():
        print(f"ERROR: Directory not found: {knowledge_path}")
        return knowledge_base
    
    for file_path in knowledge_path.glob("*.md"):
        doc_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            knowledge_base[doc_id] = content
            print(f"  Loaded: {doc_id} ({len(content)} characters)")
    
    return knowledge_base


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Convert text to a vector embedding using OpenAI.
    """
    text = text.replace("\n", " ")  # Clean up text
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    """
    # Convert to numpy arrays
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Calculate dot product
    dot_product = np.dot(a, b)
    
    # Calculate magnitudes
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    # Handle edge case
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (magnitude_a * magnitude_b)


def create_embeddings_cache(knowledge_base: dict) -> dict:
    """
    Create embeddings for all documents and cache them.
    """
    print("Creating embeddings...\n")
    
    embeddings_cache = {}
    
    for doc_id, content in knowledge_base.items():
        print(f"  Embedding: {doc_id}...")
        embedding = get_embedding(content)
        embeddings_cache[doc_id] = embedding
        print(f"    → {len(embedding)} dimensions")
    
    return embeddings_cache


def semantic_search(query: str, knowledge_base: dict, embeddings_cache: dict, top_k: int = 2) -> list:
    """
    Find most relevant documents using cosine similarity.
    """
    print(f"\nSearching for: '{query}'")
    
    # Convert query to embedding
    query_embedding = get_embedding(query)
    
    # Calculate similarity with each document
    similarities = []
    print("Similarity scores:")
    
    for doc_id, content in knowledge_base.items():
        doc_embedding = embeddings_cache[doc_id]
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_id, content, similarity))
        print(f"  - {doc_id}: {similarity:.4f}")
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Top {top_k} documents selected\n")
    return similarities[:top_k]


def rag_query(user_question: str, knowledge_base: dict, embeddings_cache: dict) -> str:
    """
    Complete RAG pipeline with semantic search.
    """
    # Search using cosine similarity
    relevant_docs = semantic_search(user_question, knowledge_base, embeddings_cache)
    
    if not relevant_docs:
        return "I don't have information about that."
    
    # Build context
    context = "\n\n".join([doc[1] for doc in relevant_docs])
    
    # Create prompt
    prompt = f"""You are a helpful bookstore assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain the information, say so politely.

Context:
{context}

Question: {user_question}

Answer:"""
    
    # Generate answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful bookstore assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content


def main():
    """
    Test the complete RAG system with semantic search
    """
    print("=" * 70)
    print("Bookstore RAG Assistant - Solution")
    print("=" * 70)
    
    # Load documents
    knowledge_base = load_documents("bookstore_knowledge")
    
    if not knowledge_base:
        print("ERROR: No documents loaded.")
        return
    
    # Create embeddings cache
    embeddings_cache = create_embeddings_cache(knowledge_base)
    
    # Test questions
    questions = [
        "What science fiction books do you have?",
        "Tell me about mystery novels",
        "What children's books are available?",
    ]
    
    print("\n" + "="*70)
    print("Testing RAG queries:")
    print("="*70)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = rag_query(question, knowledge_base, embeddings_cache)
        print(f"Answer: {answer}")
        print("-" * 70)


if __name__ == "__main__":
    main()
