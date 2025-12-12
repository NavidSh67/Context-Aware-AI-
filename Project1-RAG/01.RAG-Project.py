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

    knowledge_base = {}
    knowledge_path = Path(__file__).parent / knowledge_dir

    # /home/user/Project1  --> /home/user/Project1/knowledge

    if not knowledge_path.exists():
        print(f"Error: Knowledge directory not found: {knowledge_path}")
        return knowledge_base
    
    for file_path in knowledge_path.glob("*.md"):
        doc_id = file_path.stem   # remote_work_policy.md  --> remote_work_policy
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            knowledge_base[doc_id] = content  # {key1:value1, key2:value2, ...}  --> {doc_id:content}
    print(f"Loaded: {doc_id} ({len(content)} characters)")

    return knowledge_base




def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:

    text = text.replace("\n", " ")

    response = client.embeddings.create(input = [text], model = model)

    return response.data[0].embedding

    # response = {"object": "list",
    #             "data": [ 
    #                 {
    #                     "object": "embedding"
    #                     "index"
    #                     "embedding" : [0.00232432, 0.009343,...]
    #                 }
    #             ],
    #             "model":"text-embedding-3-small",
    #              "usage":
    #               .... }



def cosine_similarity(vec1: list, vec2: list) -> float:

    """
    Formula: cos(θ) = (A.B) / (||A||×||B||)

    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a,b)

    """
    a = [1,2,3]
    b = [4,5,6]
    dot = (1*4) + (2*5) + (3*6) = 32
    """

    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    
    return dot_product / (magnitude_a * magnitude_b)

    """
    1.0  → Identical (0° angle)
    0.9  → Very similar (25° angle)
    0.7  → Related (45° angle)
    0.5  → Somewhat related (60° angle)
    0.0  → Unrelated (90° angle - perpendicular)

    """

def create_embeddings_cache(knowledge_base: dict) -> dict:

    print("\n" + "="*70)
    print("STEP 2: Creating embeddings for all documents")
    print("="*70)
    print("Converting each document to a 1536-dimensional vector...\n")

    embeddings_cache = {}

    for doc_id, content in knowledge_base.items():
        print(f" Creating embedding for: {doc_id} ...")
        embedding = get_embedding(content)
        embeddings_cache [doc_id] = embedding
        print(f" > Vector with {len(embedding)} dimensions")

    print(f"\nCreated {len(embeddings_cache)} embeddings (cashe in memory)")
    
    """
    {key: content (text)}
    embeddings_cache = {'remote work policy': [0.023, -0.0143, ...]}
    """
    return embeddings_cache

    """
    Without cache
    Query1 > Create embeddings > Serach > Answer
    Query2 > Create embeddings > Serach > Answer

    With Cashe

    Steup: Create embedding once > cashe
    Query1 >Search (use cashe) > Answer
    Query2 >Search (use cashe) > Answer

    """

def semantic_search(query: str, knowledge_base: dict, embeddings_cache: dict, top_k: int = 2) -> list:

    print("\n" + "="*70)
    print("STEP 3: Semantic search with cosine similarity")
    print("="*70)

    query_embedding = get_embedding(query)

    similarities = []
    print(f" Calculating cosine similarity with each document:")

    for doc_id, content in knowledge_base.items():
        doc_embedding = embeddings_cache[doc_id]
        similarity = cosine_similarity(query_embedding,doc_embedding)
        similarities.append((doc_id, content, similarity))
        print(f"  -{doc_id}: {similarity: .4f}")

    # Sort by similarities (highest first)
    similarities.sort(key= lambda x: x[2], reverse=True)
    print(f" \n Top {top_k} most relevent documents selected")

    # (doc_id, content, similarity_score)
    
    return similarities[:top_k]

    """
    Query: What is the remote work policy?

    Calculating cosine similarity:
    - remote_work_policy: 0.8745  ← High similarity!
    - health_benefits: 0.6234
    - vacation_policy: 0.5891
    - office_locations: 0.4512
    - product_nexus: 0.3928      ← Low similarity
    """


def rag_query(user_question, knowledge_base, embeddings_cache):

    """
    Complete RAG pipeline with semantic search.
    
    Steps:
    1. Search using cosine similarity
    2. Build context from top results  
    3. Augment prompt with context
    4. Generate answer with OpenAI
    """
    # Step1 Search for relevant documents
    relevant_docs = semantic_search(user_question, knowledge_base, embeddings_cache)

    if not relevant_docs:
        return "I dont have information about that in the knowledgebase"
    
    #Step2 Build context from retrieved documents
    context_parts = []
    for doc_id, content, similarity in relevant_docs:
        context_parts.append(f"[Document: {doc_id}]\n{content}")

    context = "\n\n---\n\n".join(context_parts)

    """
    [Document: remote_work_policy]
    Tech corp allows employees to work remotely ...

    ---

    [Document: vacation_policy]
    Full time employees to work ...

    ---
    """
    #Step 3: Prompt Augmentation 
    prompt = f""" Answer the question based ONLY on the provided context.

    Context:
    {context}

    Question: {user_question}

    Answer:"""

    #Step4 Generating answer

    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system","content":"You are helpful assistant"},
            {"role": "user","content":prompt}
        ],
        temperature = 0.7,
        # this control the randomness/creativity   0=very focused and deterministic/ 1= Most creative
        max_tokens = 300
    )

    return response.choices[0].message.content

    # response(id, choices,model, usage,...)
    # choices (index, logprobs, message,...)
    # message (content, role, function_calls,...)


def main():
    """
    Run example RAG queries with semantic search
    """
    print("=" * 70)
    print("RAG Assistant - Semantic Search with Embeddings")
    print("=" * 70)

    # Step1: Load documents
    knowledge_base = load_knowledge_base("knowledge")
    if not knowledge_base:
        print("Error: No documents loaded")
        return
    
    #Step 2: Create embedding (one-time)
    embeddings_cache = create_embeddings_cache(knowledge_base)

    # Ask question
    questions = [
        "what is the remote work policy?",
        "How much does Nexus Cloud Platform cost?",
        "What health benefits do employees get?"
    ]

    for question in questions:
        answer = rag_query(question,knowledge_base,embeddings_cache)
        print(f"\n{'='*70}")
        print(f"ANSWER: {answer}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()


