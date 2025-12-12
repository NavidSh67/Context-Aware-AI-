# Context-Aware AI Systems Course: RAG, CAG, and KAG

A comprehensive educational repository for learning Retrieval-Augmented Generation (RAG), Cache-Augmented Generation (CAG), and Knowledge-Augmented Generation (KAG) systems.

## 📚 Course Overview

This course teaches you how to build intelligent AI systems that leverage external knowledge sources. You'll progress from basic RAG concepts to advanced KAG implementations through a structured, hands-on approach.

## 🎯 Learning Path

### **Module 1: RAG (Retrieval-Augmented Generation)**
Build AI systems that search and retrieve relevant information from documents.

#### Conceptual Videos:
1. Introduction to RAG
2. Data Source and Document Chunking
3. Embedding Models Explained
4. Vector Store & Semantic Search
5. Context Generation and Prompt Augmentation
6. RAG Pipeline Architecture

#### Hands-On Project:
- **Project 1**: [`Project1-RAG/01.RAG-Project.py`](Project1-RAG/01.RAG-Project.py)
  - Load company policy documents
  - Create embeddings with OpenAI
  - Implement semantic search with cosine similarity
  - Generate context-aware answers
- **Exercise**: [`Project1-RAG/exercise.py`](Project1-RAG/exercise.py) - Build your own RAG assistant
- **Solution**: [`Project1-RAG/solution.py`](Project1-RAG/solution.py)

---

### **Module 2: CAG (Cache-Augmented Generation)**
Optimize costs and latency by caching repeated context in system prompts.

#### Conceptual Videos:
1. Introduction to CAG

#### Hands-On Project:
- **Project 2**: [`Project2-CAG/01.CAG-Project2.py`](Project2-CAG/01.CAG-Project2.py)
  - Load knowledge base into single context
  - Implement prompt caching
  - Compare token usage (first query vs cached queries)
  - Demonstrate 50% cost savings on cached tokens
- **Exercise**: [`Project2-CAG/exercise.py`](Project2-CAG/exercise.py) - Movie recommendation system with CAG
- **Solution**: [`Project2-CAG/solution.py`](Project2-CAG/solution.py)

---

### **Module 3: KAG (Knowledge-Augmented Generation)**
Combine structured knowledge graphs with unstructured documents for superior AI reasoning.

#### Conceptual Videos:
1. Introduction to KAG
2. Knowledge Graph
3. KAG Workflow and Comparison with RAG

#### Hands-On Project:
- **Project 3**: [`Project3-KAG/01.KAG-Project3.py`](Project3-KAG/01.KAG-Project3.py)
  - Load structured knowledge graph (JSON)
  - Load unstructured documents (Markdown)
  - Extract entities from user questions
  - Query graph for structured facts
  - Search documents for detailed context
  - Fuse both knowledge sources
  - Generate comprehensive answers
- **Exercise**: [`Project3-KAG/exercise.py`](Project3-KAG/exercise.py) - Music knowledge system with KAG
- **Solution**: [`Project3-KAG/solution.py`](Project3-KAG/solution.py)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
python --version
```

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd Context-Aware

# Create and activate virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Run a project
python Project1-RAG/01.RAG-Project.py
```

---

## 📂 Repository Structure

```
Context-Aware/
│
├── Project1-RAG/
│   ├── 01.RAG-Project.py          # Main RAG implementation
│   ├── exercise.py                # Student exercise
│   ├── solution.py                # Complete solution
│   ├── knowledge/                 # Sample documents
│   │   ├── remote_work_policy.md
│   │   ├── health_benefits.md
│   │   └── vacation_policy.md
│   └── SPEAKER_NOTES_RAG.md      # Teaching guide
│
├── Project2-CAG/
│   ├── 01.CAG-Project2.py         # Main CAG implementation
│   ├── exercise.py                # Movie recommendation exercise
│   ├── solution.py                # Complete solution
│   ├── knowledge/                 # CloudTech knowledge base
│   │   ├── company_overview.md
│   │   ├── pricing_plans.md
│   │   ├── support_policy.md
│   │   └── data_centers.md
│   └── movies/                    # Exercise knowledge base
│       ├── action_movies.md
│       ├── comedy_movies.md
│       ├── scifi_movies.md
│       └── drama_movies.md
│
├── Project3-KAG/
│   ├── 01.KAG-Project3.py         # Main KAG implementation
│   ├── generate_graph.py          # Auto-generate knowledge graphs
│   ├── exercise.py                # Music knowledge exercise
│   ├── solution.py                # Complete solution
│   ├── knowledge/                 # Movie documents
│   │   ├── nolan_movies.md
│   │   └── tarantino_movies.md
│   ├── movie_graph.json           # Pre-built knowledge graph
│   ├── SPEAKER_NOTES_KAG.md      # Teaching guide
│   └── LIVE_CODING_SPEAKER_NOTES.md  # Live coding guide
│
└── .env                           # API keys (not in repo)
```

---

## Key Concepts Comparison

| Feature | RAG | CAG | KAG |
|---------|-----|-----|-----|
| **Knowledge Source** | Documents | Documents | Graph + Documents |
| **Search Method** | Semantic similarity | No search (all cached) | Graph query + Semantic search |
| **Best For** | Dynamic content | Static, frequently-accessed content | Complex reasoning with relationships |
| **Cost Efficiency** | Moderate | High (50% savings on cache) | Moderate |
| **Response Quality** | Good | Good | Excellent (structured + unstructured) |
| **Setup Complexity** | Low | Low | Medium (requires graph) |

---

## 🛠️ Common Issues & Solutions

### Issue: "Module not found"
```bash
pip install openai python-dotenv
```

### Issue: "API key not found"
Create a `.env` file with:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Issue: "Knowledge directory not found"
Make sure you're running from the project root:
```bash
python Project1-RAG/01.RAG-Project.py  # ✅ Correct
cd Project1-RAG && python 01.RAG-Project.py  # ❌ Wrong
```

---

## 📊 Learning Outcomes

After completing this course, students will be able to:

✅ Understand the architecture of RAG, CAG, and KAG systems  
✅ Implement semantic search with embeddings and cosine similarity  
✅ Optimize costs using prompt caching strategies  
✅ Build and query knowledge graphs  
✅ Fuse structured and unstructured knowledge sources  
✅ Choose the right technique for different use cases  
✅ Deploy production-ready context-aware AI systems  

---

## Additional Resources

- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Prompt Caching Guide](https://platform.openai.com/docs/guides/prompt-caching)
- [Knowledge Graphs Introduction](https://en.wikipedia.org/wiki/Knowledge_graph)

---

## Contributing

This is an educational repository. If you find issues or have suggestions:
1. Open an issue describing the problem
2. Submit a pull request with fixes
3. Share your custom exercises and knowledge bases

---

## About the Instructor

**Navid Shirzadi** - PhD in Engineering with a decade of experience in data science and AI. Passionate about making complex AI concepts accessible through hands-on, practical learning approaches.

---

**Happy Learning! 🚀**