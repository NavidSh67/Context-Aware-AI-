"""
Project 3: KAG - Knowledge Augmented Generation

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

def generate_graph_from_documents(documents: dict) -> dict:
    """
    OPTIONAL: Auto-generate knowledge graph from documents using AI
    
    """
    print("\n" + "="*70)
    print("Generating Knowledge Graph from Documents...")
    print("="*70)
        
        return {"entities": {}, "relationships": []}


def load_knowledge_graph(graph_file: str) -> dict:
    """
    STEP 1A: Load structured knowledge (facts and relationships)
    
    Graph structure:
    - ENTITIES (the things/nouns):
      {"Inception": {"type": "Movie", "year": 2010},
       "Christopher_Nolan": {"type": "Person", "nationality": "British"}}
    
    - RELATIONSHIPS (edges connecting entities):
      [{"subject": "Christopher_Nolan", "predicate": "DIRECTED", "object": "Inception"}]
      Format: subject (node1) → predicate (edge/verb) → object (node2)
      Think: Christopher_Nolan --DIRECTED--> Inception
    """
    print("\n" + "="*70)
    print("STEP 1A: Loading Knowledge Graph (Structured Facts)")
    print("="*70)
    
    return {"entities": entities, "relationships": relationships}

def load_documents(knowledge_dir: str = "knowledge") -> dict:
    """
    STEP 1B: Load unstructured documents (detailed text)
    Same as RAG!
    """
    print("\n" + "="*70)
    print("STEP 1B: Loading Documents (Unstructured Text)")
    print("="*70)
    
    documents = {}
    knowledge_path = Path(__file__).parent / knowledge_dir
    
    if not knowledge_path.exists():
        print(f"Error: Knowledge directory not found: {knowledge_path}")
        return documents
    
    for file_path in knowledge_path.glob("*.md"):
        doc_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents[doc_id] = content
            print(f"Loaded: {doc_id}")
    
    return documents

def extract_entities_from_question(question: str, graph_data: dict) -> list:
    """
    STEP 2: Find which entities from the graph are mentioned in the question
    
    """

    # Also check common variations
      
    return found_entities


def get_facts_from_graph(entity_ids: list, graph_data: dict) -> str:
    """
    STEP 3: Get structured facts about the entities
 
    """

    return "\n".join(facts) if facts else "No specific facts found."


def get_embedding(text: str) -> list:
    """Get embedding vector (same as RAG)"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate similarity (same as RAG)"""
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def search_documents(query: str, documents: dict, top_k: int = 2) -> list:
    """
    STEP 4: Search documents for relevant text (same as RAG)
    """
    query_embedding = get_embedding(query)
    similarities = []
    
    for doc_id, content in documents.items():
        doc_embedding = get_embedding(content)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_id, content, similarity))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


def kag_query(user_question: str, graph_data: dict, documents: dict) -> str:
    """
    STEP 5: Complete KAG Pipeline
    
    """
    print("\n" + "="*70)
    print("STEP 2: Extracting Entities from Question")
    print("="*70)
    
    # Extract entities
    
    print("\n" + "="*70)
    print("STEP 3: Getting Facts from Knowledge Graph")
    print("="*70)
    
    # Get structured facts
    
    print("\n" + "="*70)
    print("STEP 4: Searching Documents for Details")
    print("="*70)
    
    # Search documents

    print("\n" + "="*70)
    print("STEP 5: Fusing Knowledge + Generating Answer")
    print("="*70)
    
    # Fuse both knowledge sources
    
    # Generate answer

    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a movie expert assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content


def main():
    """
    Simple KAG demonstration
    """
    print("="*70)
    print("Movie Knowledge Assistant - KAG (Knowledge Augmented Generation)")
    print("="*70)
    
    # Step 1: Load documents first
    
    # Step 2: Load or generate graph
    
    print("\n" + "="*70)
    print("Knowledge Sources Ready!")
    print("="*70)
    print(f"✓ Graph: {len(graph_data['entities'])} entities, {len(graph_data['relationships'])} relationships")
    print(f"✓ Documents: {len(documents)} files")
    
    # Step 3: Ask questions
    questions = [
        "What sci-fi movies did Christopher Nolan direct?",
        "Tell me about Inception's critical reception"
    ]  

    for question in questions:
        answer = kag_query(question, graph_data, documents)
        print(f"\n{'='*70}")
        print(f"ANSWER: {answer}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
