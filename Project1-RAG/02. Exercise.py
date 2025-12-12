"""
Exercise: Build Your Own RAG System with Semantic Search

Your task: Complete the functions below to create a RAG system for a bookstore.
This follows the EXACT same pattern as example.py, but with simpler data.

Steps you'll implement:
1. Load documents from files
2. Get embeddings (convert text to vectors)
3. Calculate cosine similarity
4. Create embeddings cache
5. Semantic search
6. RAG query

Try each function one at a time. Test as you go!
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
    TODO: Load all .md files from the knowledge directory.
    
    Steps:
    1. Create path: Path(__file__).parent / knowledge_dir
    2. Check if path exists
    3. Loop through all .md files: knowledge_path.glob("*.md")
    4. For each file:
       - Get doc_id from file_path.stem (filename without extension)
       - Read content with open(file_path, 'r', encoding='utf-8')
       - Store in dictionary: knowledge_base[doc_id] = content
    5. Return the dictionary
    
    Hint: Copy from RAG-Project.py and modify the directory name!
    """
    knowledge_base = {}
    
    # YOUR CODE HERE
    
    return knowledge_base


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    TODO: Convert text to a vector embedding.
    
    Steps:
    1. Clean the text: text = text.replace("\n", " ")
    2. Call OpenAI API: client.embeddings.create(input=[text], model=model)
    3. Return the embedding: response.data[0].embedding
    
    Hint: This is almost identical to example.py!
    """
    # YOUR CODE HERE
    pass


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    TODO: Calculate cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Steps:
    1. Convert to numpy arrays: a = np.array(vec1), b = np.array(vec2)
    2. Calculate dot product: np.dot(a, b)
    3. Calculate magnitudes: np.linalg.norm(a), np.linalg.norm(b)
    4. Return dot_product / (magnitude_a * magnitude_b)
    5. Handle edge case: if either magnitude is 0, return 0.0
    
    Hint: This is the MOST IMPORTANT function - it's the math you learned!
    """
    # YOUR CODE HERE
    pass


def create_embeddings_cache(knowledge_base: dict) -> dict:
    """
    TODO: Create embeddings for all documents and cache them.
    
    Steps:
    1. Create empty dict: embeddings_cache = {}
    2. Loop through knowledge_base.items()
    3. For each doc_id and content:
       - Call get_embedding(content)
       - Store in cache: embeddings_cache[doc_id] = embedding
    4. Return the cache
    
    Hint: This is a simple loop calling get_embedding for each document.
    """
    embeddings_cache = {}
    
    # YOUR CODE HERE
    
    return embeddings_cache


def semantic_search(query: str, knowledge_base: dict, embeddings_cache: dict, top_k: int = 2) -> list:
    """
    TODO: Find most relevant documents using cosine similarity.
    
    Steps:
    1. Convert query to embedding: query_embedding = get_embedding(query)
    2. Create empty list: similarities = []
    3. Loop through knowledge_base.items()
    4. For each doc:
       - Get cached embedding: embeddings_cache[doc_id]
       - Calculate similarity: cosine_similarity(query_embedding, doc_embedding)
       - Append to list: (doc_id, content, similarity)
    5. Sort by similarity (highest first): similarities.sort(key=lambda x: x[2], reverse=True)
    6. Return top k results: similarities[:top_k]
    
    Hint: This is where you use your cosine_similarity function!
    """
    # YOUR CODE HERE
    pass


def rag_query(user_question: str, knowledge_base: dict, embeddings_cache: dict) -> str:
    """
    TODO: Complete RAG pipeline.
    
    Steps:
    1. Call semantic_search to find relevant documents
    2. Build context: combine document contents with "\n\n".join()
    3. Create prompt with context and question
    4. Call OpenAI chat API
    5. Return the answer
    
    Hint: Follow the same structure as example.py!
    """
    # YOUR CODE HERE
    pass


def main():
    """
    Test your RAG system - follows same pattern as example.py
    """
    print("=" * 70)
    print("Bookstore RAG Assistant - Exercise")
    print("=" * 70)
    
    # STEP 1: Load documents
    knowledge_base = load_documents("bookstore_knowledge")
    
    if not knowledge_base:
        print("ERROR: No documents loaded. Check bookstore_knowledge/ directory.")
        print("Hint: You need to create some .md files in that folder first!")
        return
    
    print(f"Loaded {len(knowledge_base)} documents\n")
    
    # STEP 2: Create embeddings cache
    print("Creating embeddings for all documents...")
    embeddings_cache = create_embeddings_cache(knowledge_base)
    
    if not embeddings_cache:
        print("ERROR: Embeddings not created. Check create_embeddings_cache function.")
        return
    
    print(f"Created {len(embeddings_cache)} embeddings\n")
    
    # Test questions
    questions = [
        "What science fiction books do you have?",
        "Tell me about mystery novels",
        "What children's books are available?",
    ]
    
    print("="*70)
    print("Testing RAG queries:")
    print("="*70)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = rag_query(question, knowledge_base, embeddings_cache)
        if answer:
            print(f"Answer: {answer}")
        else:
            print("ERROR: No answer generated. Check rag_query function.")
        print("-" * 70)


if __name__ == "__main__":
    main()
