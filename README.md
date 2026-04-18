# Advanced RAG Pipeline

An Advanced Retrieval-Augmented Generation pipeline that combines **Hybrid Search** (BM25 + Vector) with **Cross-Encoder Re-ranking** to deliver high-quality, context-grounded answers from your documents.

## Architecture

```
PDF Documents → Chunking → Embeddings → ChromaDB (Dense)
                                       → BM25 Index (Sparse)

User Query → Hybrid Search (BM25 + Vector) → Cross-Encoder Re-ranking → Gemini LLM → Answer
```

## Key Features

- **Hybrid Retrieval** — Combines keyword precision (BM25) with semantic understanding (Vector Search)
- **Cross-Encoder Re-ranking** — Scores and re-orders retrieved chunks for maximum relevance
- **Grounded Generation** — Gemini 1.5 Flash answers strictly from re-ranked context

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your API key
# Edit .env and set: GOOGLE_API_KEY=your_key_here

# Add documents
# Place your PDF files in the data/ folder

# Run
python app.py
```

## Project Structure

```
├── data/               # Your PDF documents
├── vector_store/       # ChromaDB persistence + BM25 index
├── src/
│   ├── config.py       # Environment & model configuration
│   ├── ingestion.py    # Document loading, chunking, embedding
│   ├── retrieval.py    # Hybrid search + re-ranking logic
│   └── generation.py   # LLM prompt & QA chain
├── app.py              # CLI entry point
└── requirements.txt
```

## Models Used

| Component | Model |
|-----------|-------|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini 1.5 Flash |
