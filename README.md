# Learnora — AI-Powered Personalized Learning Engine

<p align="center">
  <img src="assets/learnora-ui.png" width="900" alt="Learnora interface preview" />
</p>

<p align="center">
  <strong>Semantic retrieval + AI orchestration + metadata-aware ranking for personalized learning paths.</strong>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#engineering-decisions">Engineering decisions</a> ·
  <a href="#local-setup">Run locally</a>
</p>

---

## Why I built Learnora

Learning resources are spread across platforms and are rarely organized around the learner's actual goal, background, or prerequisite knowledge. Learnora explores a different approach: retrieve resources by **semantic intent**, then use their metadata and AI-assisted sequencing to turn those results into a structured learning path.

The project combines a data/embedding pipeline, vector retrieval, backend APIs, and a React interface rather than treating the recommendation model as a standalone notebook experiment.

## What it does

A learner enters a goal such as:

> `I want to learn machine learning for sports analytics.`

Learnora then:

1. Converts the query into an embedding with a SentenceTransformer model.
2. Searches a FAISS index containing 16,000+ learning resources.
3. Retrieves candidate resources using semantic similarity rather than simple keyword matching.
4. Uses metadata such as difficulty, prerequisites, credibility, content type, and source to structure the results.
5. Can use Gemini to sequence retrieved resources into learning phases and identify missing prerequisites.
6. Returns structured data that the React frontend renders as a learning experience.

---

## Product preview

### Learning interface

<p align="center">
  <img src="assets/learnora-ui.png" width="850" alt="Learnora learning interface" />
</p>

### Semantic retrieval

<p align="center">
  <img src="assets/semantic-search.png" width="700" alt="Learnora semantic search" />
</p>

### Generated roadmap

<p align="center">
  <img src="assets/roadmap.png" width="700" alt="Learnora generated roadmap" />
</p>

---

## Architecture

```text
                    ┌─────────────────────┐
                    │   Learning Sources  │
                    │ arXiv / YouTube /   │
                    │ Kaggle / Medium     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Ingestion & Metadata│
                    │  processing         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SentenceTransformer │
                    │     embeddings      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FAISS index     │
                    │ semantic retrieval  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌─────────────────┐   ┌──────────────────┐
          │ Metadata-aware  │   │ AI orchestration │
          │ ranking/filter  │   │ Gemini sequencing│
          └────────┬────────┘   └────────┬─────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │    Flask REST API   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React frontend    │
                    │ search / chat / UI  │
                    └─────────────────────┘
```

## How it works

### 1. Semantic search

The backend loads a SentenceTransformer model, encodes the user's query, and searches the persisted FAISS index. Retrieved records are joined with their stored metadata and assigned a similarity score.

### 2. Metadata-aware ranking

Retrieval is not the only signal. Learnora stores structured fields including:

- difficulty level
- prerequisites
- credibility score
- content type
- source
- labels

These fields can be used to make recommendations more useful than raw nearest-neighbour results.

### 3. AI-assisted roadmap generation

The AI orchestration layer can pass retrieved resources and the learner profile to Gemini to organize them into remedial, beginner, intermediate, and advanced phases. When prerequisites are identified as missing, the system can query an external search API for additional resources.

### 4. Application layer

The Flask backend exposes endpoints for health checks, semantic search, roadmap generation, and fishbone-style roadmap data. The React frontend consumes the API and renders the resulting learning experience.

---

## Engineering decisions

### Semantic retrieval instead of keyword matching

Learning queries often express intent rather than exact resource titles. Embedding-based retrieval allows the system to compare the meaning of a query with resource representations.

### FAISS for local vector retrieval

The project uses a persisted FAISS index so that retrieval can happen locally without recomputing embeddings for the full resource collection on every request.

### Metadata as a first-class part of recommendations

Similarity alone does not answer questions such as "Is this appropriate for a beginner?" or "What should I learn first?" Storing prerequisites, difficulty, credibility, and content type provides additional signals for ranking and sequencing.

### Separate retrieval from generation

The system keeps semantic retrieval and AI-assisted sequencing as distinct stages. This makes it possible to inspect the retrieved evidence before asking the LLM to organize it into a learning path.

---

## Key technical components

| Layer | Implementation |
|---|---|
| Frontend | React, React Router, Tailwind CSS |
| Backend | Python, Flask, REST APIs |
| Semantic retrieval | SentenceTransformers, FAISS |
| AI orchestration | Gemini API |
| ML stack | PyTorch, Hugging Face Transformers |
| Data | Structured JSON metadata + embedding index |
| Authentication | Auth0 integration in the frontend |

## Repository structure

```text
Learnora-AI_Engine/
├── assets/                 # UI, semantic search and roadmap visuals
├── backend/
│   ├── app.py              # Flask application and REST endpoints
│   ├── inference.py        # SentenceTransformer + FAISS retrieval
│   ├── ai_orchestration.py # LLM-assisted roadmap sequencing
│   ├── roadmap.py          # Roadmap construction
│   ├── datasets/           # Metadata and vector index
│   └── requirements.txt
├── frontend/               # React application
└── README.md
```

---

## API surface

The Flask backend currently exposes endpoints including:

```text
GET /api/health
GET /api/search?q=<query>&k=<top_k>
GET /api/roadmap?q=<query>&k=<top_k>&steps=<max_steps>
GET /api/fishbone?q=<query>&k=<top_k>
```

Example search request:

```bash
curl "http://localhost:5001/api/search?q=machine%20learning%20for%20sports%20analytics&k=10"
```

---

## Local setup

### Backend

```bash
git clone https://github.com/Advait-19/Learnora-AI_Engine.git
cd Learnora-AI_Engine/backend
pip install -r requirements.txt
python app.py
```

The backend runs on `http://localhost:5001` in the current application configuration.

### Frontend

In a second terminal:

```bash
cd Learnora-AI_Engine/frontend
npm install
npm start
```

The application expects the required model, metadata, vector index, and API credentials to be available in the configured project paths/environment.

---

## What I learned building it

Learnora started as a recommendation-system problem but became an exercise in building an end-to-end AI application. The most useful lessons were around the parts surrounding the model: preparing heterogeneous data, designing metadata, making retrieval inspectable, connecting retrieval to an application API, and handling the practical gap between a model pipeline and a usable product.

---

## Current direction

Potential next steps include:

- adaptive recommendations based on learner feedback
- persistent learner memory
- richer LLM-assisted roadmap refinement
- multimodal resource understanding
- graph-based prerequisite mapping

---

## Author

**Advait Gupta**  
Computer Science Engineering · KIIT

Interested in building practical AI systems across semantic retrieval, recommendation systems, LLM applications, and software engineering.

[GitHub](https://github.com/Advait-19) · [Learnora repository](https://github.com/Advait-19/Learnora-AI_Engine)
