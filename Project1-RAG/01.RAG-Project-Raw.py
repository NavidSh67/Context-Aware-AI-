"""
Project 1: RAG with Semantic Search (OpenAI Embeddings)

"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_knowledge_base(knowledge_dir: str = "knowledge") -> dict:
    """
    STEP 1: PREPARE DOCUMENTS - Load from external files
    """
    print("\n" + "="*70)
    print("STEP 1: Loading documents from files")
    print("="*70)


    return knowledge_base




def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:

    return response.data[0].embedding



def cosine_similarity(vec1: list, vec2: list) -> float:

    
    return dot_product / (magnitude_a * magnitude_b)



def create_embeddings_cache(knowledge_base: dict) -> dict:

    print("\n" + "="*70)
    print("STEP 2: Creating embeddings for all documents")
    print("="*70)
    print("Converting each document to a 1536-dimensional vector...\n")


    return embeddings_cache


def semantic_search(query: str, knowledge_base: dict, embeddings_cache: dict, top_k: int = 2) -> list:

    print("\n" + "="*70)
    print("STEP 3: Semantic search with cosine similarity")
    print("="*70)
    
    return similarities[:top_k]


def rag_query(user_question, knowledge_base, embeddings_cache):

    return response.choices[0].message.content


def main():
    """
    Run example RAG queries with semantic search
    """
    print("=" * 70)
    print("RAG Assistant - Semantic Search with Embeddings")
    print("=" * 70)


if __name__ == "__main__":
    main()


