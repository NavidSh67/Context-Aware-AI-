"""
SOLUTION: CAG Movie Recommendation Assistant
"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_knowledge_base(knowledge_dir: str = "movies") -> str:
    """
    Load all movie files and combine them into ONE string
    """
    print("\n" + "="*70)
    print("STEP 1: Loading Movie Database")
    print("="*70)

    knowledge_parts = []
    knowledge_path = Path(__file__).parent / knowledge_dir

    if not knowledge_path.exists():
        print(f"Error: Movies directory not found: {knowledge_path}")
        return ""
    
    # Load all markdown files
    for file_path in sorted(knowledge_path.glob("*.md")):
        doc_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            knowledge_parts.append(f"=== {doc_id.upper().replace('_', ' ')} ===\n{content}")
            print(f"Loaded: {doc_id}")

    # Combine everything into one string
    full_context = "\n\n".join(knowledge_parts)
    print(f"\nTotal: {len(full_context)} characters loaded")
    
    return full_context


def cag_query(user_question: str, knowledge_context: str) -> str:
    """
    Send question with movie database to get recommendations
    """
    system_prompt = f"""You are a helpful movie recommendation assistant.
Answer questions using the movie database below.

MOVIE DATABASE:
{knowledge_context}

Provide specific movie recommendations with titles and brief descriptions.
Be enthusiastic and helpful!"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.8,
        max_tokens=150
    )
    
    return response.choices[0].message.content


def main():
    """
    Run the movie recommendation CAG demo
    """
    print("="*70)
    print("Movie Recommendation CAG Assistant")
    print("="*70)

    # Step 1: Load all movie data
    knowledge_context = load_knowledge_base()

    if not knowledge_context:
        print("Error: No movie data loaded")
        return
    
    print("\n" + "="*70)
    print("STEP 2: Getting Movie Recommendations")
    print("="*70)

    # Step 2: Ask questions about movies
    questions = [
        "What's a good action movie with great stunts?",
        "Can you recommend a funny comedy?",
        "I want to watch a mind-bending sci-fi film",
        "What's the highest-rated drama movie?"
    ]

    for question in questions:
        print(f"\n{question}")
        print("-"*70)

        answer = cag_query(question, knowledge_context)
        print(f"Answer: {answer}")

    print("\n" + "="*70)
    print("Demo Complete! Enjoy your movies!")
    print("="*70)


if __name__ == "__main__":
    main()
