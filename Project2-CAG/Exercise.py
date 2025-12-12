"""
EXERCISE: Build Your Own CAG Assistant
Topic: Movie Recommendations

Your task: Create a CAG system that recommends movies based on a knowledge base.

HINTS:
- Follow the same structure as 01.CAG-Project2.py
- You need 3 functions: load_knowledge_base(), cag_query(), and main()
- Use the 'movies' folder for your knowledge base
- Test with questions about different movie genres
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
    TODO: Load all movie files and combine them into ONE string
    
    HINTS:
    1. Create an empty list called knowledge_parts = []
    2. Get the path: Path(__file__).parent / knowledge_dir
    3. Check if path exists, if not return ""
    4. Loop through .md files with: for file_path in sorted(knowledge_path.glob("*.md"))
    5. Read each file and append to knowledge_parts with format:
       f"=== {doc_id.upper().replace('_', ' ')} ===\n{content}"
    6. Join all parts with "\n\n".join(knowledge_parts)
    7. Return the full_context string
    """
    
    # YOUR CODE HERE
    pass


def cag_query(user_question: str, knowledge_context: str) -> str:
    """
    TODO: Send the question with knowledge base to get an answer
    
    HINTS:
    1. Create a system_prompt that includes the knowledge_context
    2. Tell the AI it's a movie recommendation assistant
    3. Use client.chat.completions.create() with:
       - model: "gpt-4o-mini"
       - messages: system message with prompt, user message with question
       - temperature: 0.8
       - max_tokens: 150
    4. Return response.choices[0].message.content
    """
    
    # YOUR CODE HERE
    pass


def main():
    """
    TODO: Run the complete CAG demo
    
    HINTS:
    1. Print a nice header
    2. Call load_knowledge_base() and store result
    3. Check if knowledge_context is empty, if so exit
    4. Use the questions list provided below
    5. Loop through questions and:
       - Print the question
       - Call cag_query()
       - Print the answer
    6. Print completion message
    """
    
    # Questions to ask (USE THESE!)
    questions = [
        "What's a good action movie with great stunts?",
        "Can you recommend a funny comedy?",
        "I want to watch a mind-bending sci-fi film",
        "What's the highest-rated drama movie?"
    ]
    
    # YOUR CODE HERE
    pass


if __name__ == "__main__":
    main()
