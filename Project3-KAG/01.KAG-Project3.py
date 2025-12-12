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

def generate_graph_from_documents(documents: dict) -> dict:
    """
    OPTIONAL: Auto-generate knowledge graph from documents using AI
    
    """
    print("\n" + "="*70)
    print("Generating Knowledge Graph from Documents...")
    print("="*70)

    all_text = "\n\n".join([f"Documents: {doc_id}\n{content}" for doc_id, content in documents.items()])

    prompt = f"""Extract a knowledge graph from this text. Return ONLY valis JSON.

{all_text[:4000]}

Format:
{{
  "entities": {{"Entity_Name": {{"type": "Person|Movie", "property": "value"}}}},
  "relationships": [{{"subject": "Entity1", "predicate": "DIRECTED", "object": "Entity2"}}]
}}

Use underscores in names (Christopher_Nolan). Use UPPERCASE verbs (DIRECTED, ACTED_IN).
JSON:
    """

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system","content":"Extract knowledge graph as JSON only"},
            {"role": "user","content":prompt}
        ],
        temperature = 0.3,
        # this control the randomness/creativity   0=very focused and deterministic/ 1= Most creative
        max_tokens = 2000
    )

    try:
        results = response.choices[0].message.content

        if "```json" in results:
            results = results.split("```json")[1].split("```json")[0]
        elif "```" in results:
            results = results.split("```")[1].split("```")[0]

        graph_data = json.loads(results.strip())
        print(f" Generated {len(graph_data.get('entities',{}))} entities")
        print(f" Generated {len(graph_data.get('relationships',{}))} relationships")

        return graph_data
    except:
        print("Could not generate graph automatically")
        
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

    graph_path = Path(__file__).parent / "knowledge" / graph_file

    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}")
        return {"entities":{}, "relationships": []}
    
    with open(graph_path, 'r', encoding = 'utf-8') as f:
        graph_data = json.load(f)

    entities = graph_data.get("entities",{})
    relationships = graph_data.get("relationships",[])
    
    return {"entities": entities, "relationships": relationships}


def extract_entities_from_question(question: str, graph_data: dict) -> list:
    """
    STEP 2: Find which entities from the graph are mentioned in the question
    
    """
    question_lower = question.lower()
    found_entities = []

    # Check eah entity in graph

    for entity_id, entity_info in graph_data['entities'].items():
        entity_name = entity_id.replace('_',' ').lower()
        # Christopher_Nolan  --> christopher nolan

        if entity_name in question_lower:
            found_entities.append(entity_id)

    # Also check common variations
    name_map = {
        'nolan': 'Christopher_Nolan',
        'inception movie': 'Incpetion',
        'interstellar film': 'Interstellar'
    } 

    for pattern, entity_id in name_map.items():
        if pattern in question_lower and entity_id not in found_entities:
            if entity_id in graph_data['entities']:
                found_entities.append(entity_id)
      
    return found_entities


def get_facts_from_graph(entity_ids: list, graph_data: dict) -> str:
    """
    STEP 3: Get structured facts about the entities

    For each entity, get:
    - Properties (year, genre, rating,...)
    - Relationships (who directed it, etc)
    
    """

    facts = []

    for entity_id in entity_ids:
        entity_info = graph_data['entities'].get(entity_id,{})

        # Get entity properties
        fact_text = f"\n{entity_id}:\n"
        for key, value in entity_info.items():
            if isinstance(value,list):
                value = ', '.join(map(str, value))   #["PERSON" "DIRECTOR"]  ---> "PERSON", "DIRECTOR"
            fact_text += f" -{key}: {value}\n"

        # Christopher_Nolan: type: PERSON ....

        # Get Relationships    
        for rel in graph_data['relationships']:
            if rel['subject'] == entity_id:
                fact_text += f" -{rel['predicate']}: {rel['object']}\n"
            elif rel['object'] == entity_id:
                fact_text += f" -{rel['subject']}: {rel['predicate']} this\n"

        # Subject  ---> Predicate ---> Object  
        # As subject: Christopher_Nolan Directed Inception
        # As Object: Inception was directed by Christopher_Nolan
        

        facts.append(fact_text)
    
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

    entities = extract_entities_from_question(user_question, graph_data)

    if entities:
        print(f"Found entities: {','.join(entities)}")
    else:
        print("No specific entities found")
    
    print("\n" + "="*70)
    print("STEP 3: Getting Facts from Knowledge Graph")
    print("="*70)
    
    # Get structured facts

    graph_facts = get_facts_from_graph(entities, graph_data)
    print(f"Retrieved facts for {len(entities)} entities")

    
    print("\n" + "="*70)
    print("STEP 4: Searching Documents for Details")
    print("="*70)
    
    # Search documents

    relevant_docs = search_documents(user_question, documents)

    doc_context = ""
    for doc_id, content, similarity in relevant_docs:
        doc_context += f"\n[Document: {doc_id}]\n{content}\n"
        print(f"Found: {doc_id} (similarity: {similarity:.3f})")

    # Document: inception_analysis
    # "Inception is Nolan's masterpiece exploring ..."

    print("\n" + "="*70)
    print("STEP 5: Fusing Knowledge + Generating Answer")
    print("="*70)
    
    # Fuse both knowledge sources
    fused_context = f"""FACTS FROM KNOWLEDGE GRAPH:
{graph_facts}

DETAILS FROM DOCUMENTS:
{doc_context}
"""

    prompt = f"""Answer the questions using BOTH the structured facts and document details.

{fused_context}

Question: {user_question}

Provide a comprehensive answer that combines factual data with rich context
"""
    
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

    documents = load_documents()

    if not documents:
        print("Error: No documents loaded")
        return
    
    # Step 2: Load or generate graph

    graph_data = load_knowledge_graph("generated_graph.json")


    # If no graph file exist, generate it from documents
    if not graph_data['entities']:
        print("\n No graph found. Generating from documents ...")
        graph_data = generate_graph_from_documents(documents)

        #Save generated graph for future use
        if graph_data['entities']:
            graph_path = Path(__file__).parent / "knowledge" / "generated_graph.json"
            with open(graph_path, 'w', encoding = 'utf-8') as f:
                json.dump(graph_data, f, indent = 2)
            print(f"Graph saved tp {graph_path}")
                      
    if not graph_data['entities']:
        print("Error: Could not load or generate knowledge graph")
        return
    
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
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}\n")
        answer = kag_query(question, graph_data, documents)
        print(f"\n{'='*70}")
        print(f"ANSWER: {answer}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
