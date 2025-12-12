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
    
    return full_context


def cag_query(user_question: str, knowledge_context: str) -> str:
    """
    STEP 2: Send question with ALL knowledge in the system prompt
    The system prompt gets cached automatically by OpenAI!
    """
    
    return response.choices[0].message.content


def main():
    """
    Simple CAG Demo - Load once, query many times!
    """
    print("="*70)
    print("CloudTech CAG Assistant - Simple Version")
    print("="*70)
    

if __name__ == "__main__":
    main()
