# P3 — Baby RAG (Embeddings + Vector Search)

## Overview
This project implements a basic Retrieval-Augmented Generation (RAG) system that allows users to ask questions about documents. The system uses embeddings and vector search to retrieve relevant document chunks and generate grounded answers using an LLM.

## Architecture
Documents → Text Splitting → Embeddings → Vector Store → Similarity Search → LLM → Answer

## Tech Stack
- Python
- LangChain
- Ollama
- FAISS / Chroma
- Streamlit / FastAPI

## How to Run
pip install -r requirements.txt
ollama run llama3.1:8b
streamlit run app.py

## Example Use Case
Document Q&A system for PDFs, research papers, and company documents.

## Key Learning
This project demonstrates the core architecture behind RAG-based LLM systems used in industry.