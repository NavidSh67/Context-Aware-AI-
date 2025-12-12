"""
Project 2: CAG - Cache Augmented Generation (OpenAI Prompt Caching)

"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_knowledge_base(knowledge_dir: str = "knowledge") -> str:
    """
    STEP 1: Load all documents and combine them into ONE big string
    Unlike RAG (which keeps docs separate), CAG combines everything!
    """
    print("\n" + "="*70)
    print("STEP 1: Loading Knowledge Base")
    print("="*70)

    knowledge_parts = []
    knowledge_path = Path(__file__).parent / knowledge_dir

    if not knowledge_path.exists():
        print(f"Error: Knowledge directory not found: {knowledge_path}")
        return ""
    
    #Load all markdown files
    for file_path in sorted(knowledge_path.glob("*.md")):
        doc_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            knowledge_parts.append(f"=== {doc_id.upper().replace('_',' ')} ===\n{content}")
            # Example:  "=== DATA CENTERS ===\n[file content here]"
            print(f"Loaded: {doc_id}")

    # Combine everything into one string
    full_context = "\n\n".join(knowledge_parts)
    print(f"\nTotal: {len(full_context)} characters combined")
    
    return full_context


def cag_query(user_question: str, knowledge_context: str) -> str:
    """
    STEP 2: Send question with ALL knowledge in the system prompt
    The system prompt gets cached automatically by OpenAI!
    """
    system_prompt = f""" You are a helpful assistant for CloudTech Solutions.
    Answer questions using the knowledge base below.

    KNOWLEDGE BASE:
    {knowledge_context}

    Be concise and helpful!"""


    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system","content":system_prompt},
            {"role": "user","content":user_question}
        ],
        temperature = 0.7,
        # this control the randomness/creativity   0=very focused and deterministic/ 1= Most creative
        max_tokens = 200
    )
    
    return response.choices[0].message.content


def main():
    """
    Simple CAG Demo - Load once, query many times!
    """
    print("="*70)
    print("CloudTech CAG Assistant - Simple Version")
    print("="*70)

    # Step1: Load all knowledge into one string

    knowledge_context = load_knowledge_base()

    if not knowledge_context:
        print("Error: No knowledge loaded")
        return
    
    print("="*70)
    print("Step2: Asking Questions (Knowledge Gets Chached)")
    print("="*70)

    # Step2: Ask Multiple Questions

    questions = [
        "What are your pricing plans?",
        "Do you have data centers in Europe?",
        "How can I get support?"
    ]

    for question in questions:
        print(f"\n\n{question}")
        print("-"*70)

        answer = cag_query(question,knowledge_context)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
