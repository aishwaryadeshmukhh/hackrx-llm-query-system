# LLM-Based Insurance Query System

AI-powered insurance query analysis system built for HackRx 6.0 using LLMs, semantic retrieval, and Retrieval-Augmented Generation (RAG) pipelines.

---

## 📌 Overview

This project is designed to process natural-language insurance queries and retrieve relevant information from large unstructured documents such as:

- Insurance policies
- Contracts
- Emails
- PDFs
- Claim documents

The system performs semantic retrieval and LLM-based reasoning to generate structured claim decisions with supporting justifications.

---

## 🚀 Features

- 🔍 Semantic search over insurance documents
- 🧠 LLM-powered reasoning pipeline
- 📄 PDF and document ingestion
- 📌 Clause-level evidence retrieval
- 📦 Structured JSON responses
- ⚡ FastAPI backend integration
- 🧾 Context-aware decision generation

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI

### AI / ML
- LangChain
- OpenAI / Gemini APIs
- RAG Pipeline
- Vector Embeddings

### Data & Retrieval
- FAISS / Vector Database
- Document Chunking
- Semantic Search

---

## 📂 Project Workflow

1. Upload insurance-related documents
2. Convert documents into embeddings
3. Store embeddings in vector database
4. Accept natural-language user query
5. Retrieve most relevant clauses
6. Use LLM for contextual reasoning
7. Generate structured JSON response

---

## 📥 Example Query

```json
{
  "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
}
```

---

## 📤 Example Output

```json
{
  "Decision": "Yes",
  "Amount": 150000,
  "Justification": "Clause X covers knee replacements after waiting period.",
  "Supporting Clause": "Page 12, Clause X"
}
```

---

## 💡 Key Capabilities

- Handles vague and incomplete queries
- Performs contextual reasoning instead of keyword matching
- Retrieves exact supporting clauses
- Produces interpretable structured outputs
- Designed for insurance claim automation workflows

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/hackrx-llm-query-system.git
cd hackrx-llm-query-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn app:app --reload
```

---

## 🌟 Future Improvements

- Multi-document reasoning
- Better citation tracing
- Dashboard visualization
- Deployment with Docker
- Authentication & user sessions

---

## 📜 License

This project is intended for educational and hackathon purposes.
