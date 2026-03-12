# 🤖 AI Knowledge Assistant ( RAG + AutoGen Agents )

- An interactive **AI-powered research assistant** built with **Streamlit**, **AutoGen multi-agent orchestration**, and **Retrieval-Augmented Generation (RAG)**.  
- Upload research papers (PDF/DOCX), ask questions, and get **cited, concise answers** backed by local documents and live external sources from **arXiv** and **Wikipedia**.

---

## ✨ Features

### 📂 Document Ingestion & Indexing
- Upload **PDF or DOCX** research papers.
- Split into chunks, embed with **Hugging Face sentence-transformers**, and store in **FAISS Vector Store** (in-memory session state).
- Uses `chunk_size=400` with **MMR (Maximal Marginal Relevance)** retrieval to avoid duplicate chunks.

### 🤖 AutoGen Multi-Agent Orchestration
- **`Knowledge_Assistant`** — an LLM-backed AutoGen agent that plans and calls tools.
- **`User_Proxy`** — executes tool calls, manages termination condition.
- Agent automatically decides when to stop searching and synthesize a final answer.

### 🔀 Conditional Tool Routing
| State | Tools Available |
|---|---|
| File uploaded + external enabled | `search_local_docs` → `search_wiki` / `search_arxiv` |
| File uploaded only | `search_local_docs` only |
| No file uploaded | `search_wiki` + `search_arxiv` only |

### 🌐 External Knowledge (arXiv + Wikipedia)
- Fetches Wikipedia summaries and arXiv abstracts on-demand.
- Outputs are **truncated** to keep LLM input tokens minimal.

### 🔗 Clickable Citation Cards
- After every agent run, **external sources** (Wikipedia and arXiv) are displayed as color-coded clickable citation badges:
  - 🔵 **Wikipedia** — blue badge
  - 🔴 **arXiv PDF** — red badge
- Local document chunks cited as `[Local Source N]` in the answer.

### 🛡️ Token Optimization
- Compressed system prompts.
- 1-sentence Wikipedia fetch with 600-character cap.
- 1 arXiv result with 250-character abstract cap.
- MMR deduplication for FAISS retrieval.
- Per-call tool limit: agent instructed to call each tool at most once.

---

## 🏗️ Architecture

<img width="1536" height="1024" alt="AI Knowledge Assistant System Flowchart" src="https://github.com/user-attachments/assets/258d439a-1c5c-496a-8c01-60fbeec968bd" />

---

### Document Processing
- PDF/DOCX → chunked (400 chars) → embedded → stored in **FAISS DB** (session state).

### AutoGen Agents
- `Knowledge_Assistant` calls tools, synthesizes answer, ends with `TERMINATE`.
- Dynamic system message tailored based on whether local documents are available.
- `User_Proxy` executes functions and watches for termination signal.

### Knowledge Retrieval
- Local: **FAISS MMR retriever** over uploaded documents.
- External: Live **arXiv** search + **Wikipedia** page summary.
- URL-tracking closures capture every source link fetched during the agent run.

### UI
- Streamlit app with custom CSS, expandable settings panel, and sidebar downloads.
- Response box + external citation cards + local citation panels.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ShyamKumar-AI-DS/AI_Knowledge_Assisstant_3.git
cd AI_Knowledge_Assisstant_3
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM Backend | Groq API (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) |
| Agent Orchestration | **AutoGen** (`pyautogen==0.2.25`) |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | **FAISS** (in-memory) |
| Document Loaders | LangChain `PyPDFLoader`, `Docx2txtLoader` |
| External Knowledge | **arXiv API**, **Wikipedia API** |
| Frontend | **Streamlit** |
| Env Config | `python-dotenv` |

---

## 📁 Project Structure

```
AI_Knowledge_Assisstant_3/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── resources/
│   ├── Synthetic Aperture Radar.pdf    # Sample paper
│   └── Generative AI Survey.pdf       # Sample paper
└── README.md
```

---

## 📝 Example Queries

- *"What are the key achievements of Synthetic Aperture Radar?"*
- *"Summarize the scope of the SAR project"*  
- *"What are the disadvantages of Generative AI?"*
- *"What are the main components of Agentic AI workflows?"*

---

## 👨‍💻 Developed By

**Shyam Kumar**  
Upload research papers → ask questions → get AI-powered, citation-backed insights.
