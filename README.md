# Learnora — AI-Powered Personalized Learning Engine

Learnora is an AI-driven learning recommendation system that generates personalized learning roadmaps using semantic search, vector similarity retrieval, and metadata-driven ranking.

Instead of relying on keyword matching, Learnora understands the semantic intent behind user queries and dynamically structures educational content into guided learning paths.

---

## Key Highlights

- Semantic search using SentenceTransformers
- Vector similarity retrieval with FAISS
- Personalized roadmap generation
- Metadata-driven ranking and filtering
- Difficulty-aware learning progression
- Real-time structured learning path generation
- Hybrid AI pipeline combining NLP + retrieval systems

---

## Problem Statement

Modern learning platforms provide massive amounts of educational content but lack personalized structure.

Learners often struggle with:
- discovering relevant resources
- understanding prerequisite order
- identifying beginner-friendly content
- building structured learning paths

Learnora addresses this by generating dynamic, intent-aware learning roadmaps tailored to the user's learning goals.

---

## System Workflow

### Query Understanding
User queries are converted into semantic embeddings using a fine-tuned SentenceTransformer model.

### Vector Retrieval
Precomputed embeddings for 16,000+ learning resources are indexed using FAISS for fast similarity search.

### Metadata-Based Ranking
Retrieved resources are filtered and ranked using:
- difficulty level
- prerequisites
- credibility score
- content type

### Dynamic Roadmap Generation
The backend generates:
- ordered learning paths
- categorized learning branches
- prerequisite-aware progression
- visual roadmap structures

### Frontend Rendering
Structured roadmap responses are rendered through a React-based chat interface.

---

## Tech Stack

### AI / Machine Learning
- SentenceTransformers
- Hugging Face Transformers
- FAISS
- PyTorch

### Backend
- Python
- Flask
- REST APIs

### Frontend
- React
- React Router

### Data & Infrastructure
- JSON metadata pipelines
- Vector indexing
- Embedding preprocessing pipelines

---

## Architecture

```text
User Query
   ↓
React Frontend
   ↓
Flask API
   ↓
SentenceTransformer Embedding
   ↓
FAISS Vector Search
   ↓
Metadata Filtering & Ranking
   ↓
Roadmap Generation
   ↓
Structured JSON Response
   ↓
Frontend Visualization
```

---

## Core Features

- Semantic learning resource retrieval
- Personalized AI-generated learning paths
- Difficulty-based sequencing
- Prerequisite-aware recommendations
- Chat-style interaction workflow
- Categorized educational resource organization

---

## Engineering Challenges Solved

### Scaling Semantic Retrieval
Migrated from API-based embeddings to a locally fine-tuned SentenceTransformer pipeline to reduce inference costs and improve semantic consistency.

### Large-Scale Vector Indexing
Optimized FAISS indexing and embedding pipelines for efficient retrieval across 16,000+ resources.

### Roadmap Structuring Logic
Designed metadata-aware ranking logic for prerequisite alignment and progressive difficulty ordering.

---

## Project Structure

```text
frontend/
backend/
datasets/
models/
utils/
```

---

## Local Setup

```bash
# Clone repository
git clone <repo-url>

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
```

Run backend and frontend separately.

---

## Future Improvements

- Adaptive recommendation feedback loops
- User-specific learning memory
- LLM-assisted roadmap refinement
- Multi-modal resource understanding
- Graph-based prerequisite mapping

---

## Author

Advait Gupta

AI/ML enthusiast focused on intelligent systems, semantic retrieval, recommendation systems, and personalized learning technologies.
