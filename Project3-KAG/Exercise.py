"""
Exercise 3: KAG - Knowledge Augmented Generation

Your task: Complete the missing functions to build a KAG system for music recommendations.

Try to solve this yourself before looking at solution.py!
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_documents(knowledge_dir: str = "music_knowledge") -> dict:
    """
    Load unstructured documents (detailed text)
    Same as RAG!
    """
    # TODO: YOUR CODE HERE
    # Hint: Loop through .md files in music_knowledge folder
    # Return dictionary: {doc_id: content}
    pass


def generate_graph_from_documents(documents: dict) -> dict:
    """
    OPTIONAL: Auto-generate knowledge graph from documents using AI
    
    If you don't have a graph file, this function can create one!
    Uses OpenAI to extract entities and relationships from your documents.
    """
    # TODO: YOUR CODE HERE
    # Hint 1: Combine all documents into one text string
    # Hint 2: Create a prompt asking GPT to extract entities and relationships
    # Hint 3: Use gpt-4o-mini model with temperature=0.3
    # Hint 4: Parse the JSON response
    # Hint 5: Return {"entities": {}, "relationships": []}
    pass


def load_knowledge_graph(graph_file: str) -> dict:
    """
    Load structured knowledge (facts and relationships)
    
    Graph structure:
    - ENTITIES: {"The_Beatles": {"type": "Band", "genre": "Rock"}}
    - RELATIONSHIPS: [{"subject": "The_Beatles", "predicate": "RELEASED", "object": "Abbey_Road"}]
    """
    # TODO: YOUR CODE HERE
    # Hint: Load JSON file from music_knowledge folder
    # Return: {"entities": {}, "relationships": []}
    pass


def extract_entities_from_question(question: str, graph_data: dict) -> list:
    """
    Find which entities from the graph are mentioned in the question
    
    Simple approach: Check if entity names appear in question
    """
    # TODO: YOUR CODE HERE
    # Hint 1: Convert question to lowercase
    # Hint 2: Loop through all entities in graph_data['entities']
    # Hint 3: Replace underscores with spaces in entity names
    # Hint 4: Check if entity name appears in question
    pass


def get_facts_from_graph(entity_ids: list, graph_data: dict) -> str:
    """
    Get structured facts about the entities
    
    For each entity:
    - Get properties (type, genre, year, etc.)
    - Get relationships (who released what, etc.)
    """
    # TODO: YOUR CODE HERE
    # Hint 1: Loop through entity_ids
    # Hint 2: Get entity info from graph_data['entities']
    # Hint 3: Format properties as text
    # Hint 4: Find relationships where entity is subject or object
    pass


def get_embedding(text: str) -> list:
    """Get embedding vector (same as RAG)"""
    # TODO: YOUR CODE HERE
    # Hint: Use client.embeddings.create()
    pass


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate similarity (same as RAG)"""
    # TODO: YOUR CODE HERE
    # Hint: Use numpy - np.dot(), np.linalg.norm()
    pass


def search_documents(query: str, documents: dict, top_k: int = 2) -> list:
    """
    Search documents for relevant text (same as RAG)
    """
    # TODO: YOUR CODE HERE
    # Hint 1: Get query embedding
    # Hint 2: Get embedding for each document
    # Hint 3: Calculate similarity scores
    # Hint 4: Sort and return top_k
    pass


def kag_query(user_question: str, graph_data: dict, documents: dict) -> str:
    """
    Complete KAG Pipeline
    
    Steps:
    1. Extract entities from question
    2. Get facts from knowledge graph
    3. Search documents for details
    4. Fuse both knowledge sources
    5. Generate answer with OpenAI
    """
    # TODO: YOUR CODE HERE
    # Step 1: Extract entities
    
    # Step 2: Get graph facts
    
    # Step 3: Search documents
    
    # Step 4: Fuse knowledge
    
    # Step 5: Generate answer
    pass


def main():
    """
    Simple KAG demonstration
    """
    print("="*70)
    print("Music Knowledge Assistant - KAG (Knowledge Augmented Generation)")
    print("="*70)
    
    # Step 1: Load documents
    documents = load_documents()
    
    if not documents:
        print("Error: No documents loaded")
        return
    
    # Step 2: Load or generate graph
    graph_data = load_knowledge_graph("music_graph.json")
    
    # TODO: Add logic to try generated_graph.json if music_graph.json not found
    # TODO: Add logic to generate graph from documents if no graph exists
    # TODO: Add logic to save generated graph for future use
    
    if not graph_data['entities']:
        print("Error: Could not load knowledge graph")
        return
    
    print("\n" + "="*70)
    print("Knowledge Sources Ready!")
    print("="*70)
    print(f"✓ Graph: {len(graph_data['entities'])} entities, {len(graph_data['relationships'])} relationships")
    print(f"✓ Documents: {len(documents)} files")
    
    # Step 3: Ask questions
    questions = [
        "What albums did The Beatles release?",
        "Tell me about rock music in the 1970s"
    ]
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print(f"{'='*70}")
        answer = kag_query(question, graph_data, documents)
        print(f"\n{'='*70}")
        print(f"ANSWER: {answer}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()