# 🤖 AI Knowledge Assistant (CrewAI + RAG)

An interactive **AI-powered research assistant** built with **Streamlit**, **CrewAI agents**, and **Retrieval-Augmented Generation (RAG)**.  
It allows you to upload research papers (PDFs), query them, and receive **concise answers, step-by-step explanations, and summaries**.  
The app also integrates **external knowledge** from **arXiv** and **Wikipedia** to complement document understanding..

---
## ✨ Features

### 📂 PDF Ingestion & Indexing
- Upload research papers.  
- Split, embed, and store with **Chroma Vector Store**.  

### 🔍 Retrieval-Augmented QA
- Search within ingested papers using **Hugging Face embeddings**.  
- Get concise, citation-backed answers.  

### 👥 Multi-Agent CrewAI Orchestration
- **Summarizer Agent** → generates concise bullet-point summaries.  
- **Explainer Agent** → explains context step-by-step in beginner-friendly terms.  

### 🌐 External Knowledge
- Fetch related results from **arXiv** and **Wikipedia**.  

### 🎨 Custom UI
- Built with **Streamlit**, styled with **dynamic CSS**.  

---

## 🏗️ Architecture
---

<img width="1536" height="1024" alt="AI Knowledge Assistant System Flowchart" src="https://github.com/user-attachments/assets/258d439a-1c5c-496a-8c01-60fbeec968bd" />

---

### Document Processing
- PDFs → split into chunks → embeddings → stored in **Chroma**.  

### CrewAI Agents
- **Summarizer Agent**: distills document content.  
- **Explainer Agent**: teaches concepts step-by-step.  
- Sequential workflow orchestrated by **CrewAI**.  

### Knowledge Retrieval
- RAG pipeline with **Groq LLM** + **Chroma retriever**.  
- Optional external context from **arXiv & Wikipedia**.  

### UI
- Streamlit app with expandable panels, styled components, and side downloads.  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ai-knowledge-assistant.git](https://github.com/ShyamKumar-AI-DS/AI_Knowledge_Assisstant_.git)
cd AI_Knowledge_Assisstant_
