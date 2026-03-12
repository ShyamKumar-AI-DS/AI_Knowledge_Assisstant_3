# ai_knowledge_assistant_app.py

import os
import tempfile
import streamlit as st
import arxiv
import wikipedia
import autogen
from dotenv import load_dotenv

# LangChain
from langchain_groq import ChatGroq     
<<<<<<< HEAD
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
=======
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
>>>>>>> 08188d06bd81b0b686b6c6524c7490282ce2f262
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

# -------------------------
# Load LLM (Groq-based)
# -------------------------
load_dotenv()

@st.cache_resource
def get_llm():
    return ChatGroq(
        model_name="openai/gpt-oss-120b", # Or your preferred Groq model
        temperature=0.1,      # Lower temperature (0.0 - 0.2) minimizes hallucinations
        max_tokens=None,      # Let the model decide the length based on prompt
        timeout=None,
        max_retries=2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        # top_p=0.9           # Optional: Nucleus sampling to keep results focused
    )

# -------------------------
# Document ingestion → FAISS in memory
# -------------------------
def ingest_document_to_faiss(uploaded_file, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(temp_path)
    elif uploaded_file.name.endswith('.docx'):
        loader = Docx2txtLoader(temp_path)
    else:
        os.unlink(temp_path)
        st.error(f"Unsupported file format: {uploaded_file.name}")
        return

    raw_docs = loader.load()
    # Use a reasonable chunk size; too small causes duplicate chunks being returned
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    docs = text_splitter.split_documents(raw_docs)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cuda'},
    )

    if 'faiss_db' in st.session_state:
        st.session_state['faiss_db'].add_documents(docs)
    else:
        st.session_state['faiss_db'] = FAISS.from_documents(docs, hf_embeddings)

    os.unlink(temp_path)
    return st.session_state['faiss_db']

# -------------------------
# External knowledge
# -------------------------
def fetch_arxiv_papers(query, max_results=1):
    results = []
    search = arxiv.Search(query=query, max_results=max_results)
    for r in search.results():
        results.append({
            "title": r.title,
            # Extremely short summary to save tokens
            "summary": r.summary[:250] + "...",
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id
        })
    return results

def fetch_wikipedia_summary(topic, sentences=2):
    try:
        page = wikipedia.page(topic)
        summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=True, redirect=True)
        # Balance between enough content to answer and token savings
        return {"title": page.title, "summary": summary[:600], "url": page.url}
    except Exception as e:
        return {"title": topic, "summary": f"Wikipedia fetch error: {e}", "url": ""}

def add_external_results_to_faiss(external_texts, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cuda'}
    )
    docs = [Document(page_content=item["text"], metadata=item.get("meta", {})) for item in external_texts]

    if 'faiss_db' in st.session_state:
        st.session_state['faiss_db'].add_documents(docs)
    else:
        st.session_state['faiss_db'] = FAISS.from_documents(docs, hf_embeddings)
    return st.session_state['faiss_db']

<<<<<<< HEAD

def get_faiss_documents(query: str, k: int = 1) -> str:
    """Retrieve top local FAISS document for a user query."""
    if 'faiss_db' not in st.session_state:
        return "No local knowledge base uploaded yet."
    db = st.session_state['faiss_db']
    # Use MMR (Maximal Marginal Relevance) to avoid returning duplicate/similar chunks
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 10})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant documents found in the local knowledge base."

    doc_texts = []
    for i, d in enumerate(docs[:k]):
        content = getattr(d, "page_content", str(d))
        doc_texts.append(f"[Local Source {i+1}]:\n{content}")
    return "\n\n---\n\n".join(doc_texts)

def search_wikipedia(query: str) -> str:
    """Gets a summary from wikipedia for a query."""
    return str(fetch_wikipedia_summary(query))

def search_arxiv(query: str) -> str:
    """Gets relevant arxiv papers for a query."""
    return str(fetch_arxiv_papers(query))

=======


# -------------------------
# Groq-native Summarizer & Explainer
# -------------------------


# def summarize_with_groq(docs_text):
#     prompt = PromptTemplate.from_template(
#         "Summarize the following documents into 5-6 concise bullet points:\n\n{docs}"
#     )
#     llm = get_llm()
#     chain = LLMChain(llm=llm, prompt=prompt)
#     return chain.run(docs=docs_text)

# def explain_with_groq(docs_text, question):
#     prompt = PromptTemplate.from_template(
#         "Explain the following context to a beginner, step by step, and then answer the question. "
#         "End with a one-sentence summary.\n\nContext:\n{docs}\n\nQuestion: {question}"
#     )
#     llm = get_llm()
#     chain = LLMChain(llm=llm, prompt=prompt)
#     return chain.run(docs=docs_text, question=question)

def summarize_with_groq(docs_text):
    prompt = ChatPromptTemplate.from_template(
        "SYSTEM: You are a precise document parser. Your goal is to extract key insights without hallucinating details not present in the source.\n\n"
        "USER: Summarize the following context into 5-6 distinct bullet points. \n"
        "RULES:\n"
        "- Each bullet point MUST be on a new line.\n"
        "- Each bullet point MUST be entirely in **bold**.\n"
        "- Do not include any introductory text or closing remarks.\n\n"
        "CONTEXT: {docs}\n\n"
        "ASSISTANT:"
    )   
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"docs": docs_text})

def explain_with_groq(docs_text, question):
    prompt = ChatPromptTemplate.from_template(
        "SYSTEM: You are an expert analyst. Answer the question based ONLY on the provided context. "
        "If the answer is not contained within the context, state that you do not have enough information.\n\n"
        "USER:\n"
        "CONTEXT:\n{docs}\n\n"
        "QUESTION: {question}\n\n"
        "INSTRUCTIONS:\n"
        "1. Provide a step-by-step logical explanation.\n"
        "2. Directly answer the question based on those steps.\n"
        "3. Conclude with a single-sentence summary.\n\n"
        "ASSISTANT:"
    )
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"docs": docs_text, "question": question})
>>>>>>> 08188d06bd81b0b686b6c6524c7490282ce2f262


# -------------------------
# AutoGen Agent Orchestration
# -------------------------
<<<<<<< HEAD

def run_autogen_agents(question, do_external=True):
    has_local_docs = 'faiss_db' in st.session_state
    # Track external URLs fetched during this agent run
    external_sources = []

    llm_config = {
        "config_list": [
            {
                "model": "openai/gpt-oss-20b", # Using Groq's 20B OSS model
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": "https://api.groq.com/openai/v1"
            }
        ],
        "temperature": 0.1,
    }

    # Build dynamic system message based on what sources are available
    if has_local_docs and do_external:
        source_instruction = "Use search_local_docs FIRST, then supplement with search_wiki or search_arxiv if needed."
    elif has_local_docs:
        source_instruction = "Use search_local_docs to answer from uploaded documents."
    else:
        source_instruction = "No local documents available. Use search_wiki and search_arxiv to answer."

    assistant = autogen.AssistantAgent(
        name="Knowledge_Assistant",
        llm_config=llm_config,
        system_message=f"""Knowledge Assistant: {source_instruction} Call each tool at most ONCE. If a tool returns no useful data, do NOT retry with a similar query — instead use your own knowledge to answer. Provide explicit inline citations like [Wikipedia: Title], [arXiv: URL], or [Local Source 1]. Do NOT use 【1†source】 format. Be brief. End ONLY with 'TERMINATE'."""
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
    )

    # Register local FAISS tool ONLY if a document has been uploaded
    if has_local_docs:
        autogen.agentchat.register_function(
            get_faiss_documents,
            caller=assistant,
            executor=user_proxy,
            name="search_local_docs",
            description="Search local documents via FAISS."
        )

    if do_external:
        # Wrapper that captures Wikipedia URLs as they are fetched
        def search_wiki_tracked(query: str) -> str:
            result = fetch_wikipedia_summary(query)
            if result.get("url"):
                external_sources.append({"type": "Wikipedia", "title": result["title"], "url": result["url"]})
            return str(result)

        # Wrapper that captures arXiv URLs as they are fetched
        def search_arxiv_tracked(query: str) -> str:
            results = fetch_arxiv_papers(query)
            for r in results:
                if r.get("pdf_url"):
                    external_sources.append({"type": "arXiv", "title": r["title"], "url": r["pdf_url"]})
            return str(results)

        autogen.agentchat.register_function(
            search_wiki_tracked,
            caller=assistant,
            executor=user_proxy,
            name="search_wiki",
            description="Search Wikipedia."
        )
        autogen.agentchat.register_function(
            search_arxiv_tracked,
            caller=assistant,
            executor=user_proxy,
            name="search_arxiv",
            description="Search arXiv papers."
        )

    chat_res = user_proxy.initiate_chat(
        assistant,
        message=question,
        summary_method="last_msg"
    )

    final_answer = chat_res.summary.replace("TERMINATE", "").strip() if chat_res.summary else "Agent failed to respond."
    return final_answer, external_sources
=======


def run_agents(question, retriever, qa_chain):
    docs = retriever.get_relevant_documents(question) or []
    doc_texts = [getattr(d, "page_content", str(d)) for d in docs[:6]]
    documents_joined = "\n\n---\n\n".join(doc_texts)

    try:
        summary_text = summarize_with_groq(documents_joined)
        explanation_text = explain_with_groq(documents_joined, question)

    except Exception as e:
        summary_text, explanation_text = f"Groq failed: {e}", f"Groq failed: {e}"

    try:
        qa_result = qa_chain.invoke({"query": question})
        concise_answer = qa_result.get("result", "")
        source_docs = qa_result.get("source_documents", [])

    except Exception:
        concise_answer, source_docs = "", docs

    return {
        "concise_answer": concise_answer,
        "explanation": explanation_text,
        "summary": summary_text,
        "source_documents": source_docs
    }

# -------------------------
# Retrieval QA
# -------------------------

def build_retrieval_qa(k=3):
    if 'faiss_db' not in st.session_state:
        raise ValueError("No FAISS index found. Upload PDFs first.")
    db = st.session_state['faiss_db']
    retriever = db.as_retriever(search_kwargs={"k": k})
    qa_chain = get_llm | retriever
    return qa_chain, retriever
>>>>>>> 08188d06bd81b0b686b6c6524c7490282ce2f262

# -------------------------
# UI
# ------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-size: 1em;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .stTextInput input {
            border: 2px solid #4CAF50;
            border-radius: 6px;
            padding: 0.5em;
        }
        .stExpander {
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .result-box {
            background: #f9f9f9;
            padding: 1em;
            border-radius: 10px;
            margin: 0.5em 0;
            border-left: 5px solid #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    inject_custom_css()
<<<<<<< HEAD
    st.title("🤖 AI Knowledge Assistant (Autogen Agents)")
    uploaded = st.file_uploader("Upload Document(s) (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
=======
    st.title("🤖 AI Knowledge Assistant (RAG + AutoGen Agents)")
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
>>>>>>> 08188d06bd81b0b686b6c6524c7490282ce2f262
    if uploaded:
        for f in uploaded:
            ingest_document_to_faiss(f)
        st.success("✅ Document(s) ingested successfully!")

    question = st.text_input("Enter your question")
    with st.expander("⚙️ External Knowledge Settings"):
        do_external = st.checkbox("Fetch external results (arXiv + Wikipedia)", value=True)
        arxiv_max = st.slider("Number of arXiv results", 1, 10, 3)

    if st.button("🚀 Run Agent") and question.strip():
        with st.spinner("Analyzing and generating response with AutoGen..."):
            unified_answer, external_sources = run_autogen_agents(question, do_external=do_external)

<<<<<<< HEAD
        st.subheader("💡 Knowledge Assistant AI Response")
        st.markdown(f"<div class='result-box'>{unified_answer}</div>", unsafe_allow_html=True)
=======
        # st.subheader("📌 Answer")
        # st.markdown(f"<div class='result-box'>{result['concise_answer']}</div>", unsafe_allow_html=True)
>>>>>>> 08188d06bd81b0b686b6c6524c7490282ce2f262

        # Display external source citation links
        if external_sources:
            st.subheader("🔗 External Source Citations")
            # Deduplicate by URL
            seen_urls = set()
            unique_sources = [s for s in external_sources if s["url"] not in seen_urls and not seen_urls.add(s["url"])]
            for src in unique_sources:
                icon = "📖" if src["type"] == "Wikipedia" else "📄"
                badge_color = "#1a73e8" if src["type"] == "Wikipedia" else "#d62728"
                st.markdown(
                    f"<div class='result-box' style='border-left: 5px solid {badge_color}; padding: 0.6em 1em;'>"
                    f"<span style='background:{badge_color};color:white;border-radius:4px;padding:2px 8px;font-size:0.8em;font-weight:bold;'>{src['type']}</span>"
                    f"&nbsp; {icon} <a href='{src['url']}' target='_blank' style='color:{badge_color};font-weight:600;'>{src['title']}</a>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        if 'faiss_db' in st.session_state:
            st.subheader("📚 Local Exact Citations (If Applicable)")
            db = st.session_state['faiss_db']
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)
            for i, d in enumerate(docs):
                source_content = getattr(d, 'page_content', '')
                st.markdown(f"**[Local Source {i+1}]**: <div class='result-box' style='font-size: 0.9em; border-left: 3px solid #ccc;'>{source_content}</div>", unsafe_allow_html=True)

<<<<<<< HEAD
=======
        st.subheader("📚 Sources")
        for d in result["source_documents"][:5]:
            st.markdown(f"<div class='result-box'>{getattr(d,'page_content','')[:100]}...</div>", unsafe_allow_html=True)

        if "external" in st.session_state:
            for a in st.session_state["external"].get("arxiv", []):
                st.markdown(f"- [{a['title']}]({a['pdf_url']})")
            wiki = st.session_state["external"].get("wiki", {})
            if wiki:
                st.markdown(f"- **Wikipedia:** [{wiki['title']}]({wiki['url']}) — {wiki['summary']}")

>>>>>>> 08188d06bd81b0b686b6c6524c7490282ce2f262
with st.sidebar:
    st.subheader("📘 AI Knowledge Assistant")
    st.info("""
    This AI agent combines:
    - **Generative Text + RAG + Agents** and Multi-Format Document Ingestion (.pdf, .docx)
    - **arXiv + Wikipedia** for external context
    - **Exact content citations** matching user questions
    - **Autogen & FAISS** for Vector Store and Agenta orchestration
    
    Example queries:
    - Synthetic Aperture Radar (SAR) and Key Achievements
    - Generative AI and Key Achievements
    """)

    pdf_path = "./resources/Synthetic Aperture Radar.pdf"
    pdf_path1 = "./resources/Generative AI Survey.pdf"

    try:
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="📄 Download Sample Research Paper",
                data=file,
                file_name="Synthetic Aperture Radar.pdf",
                mime="application/pdf"
            )
    except Exception:
        pass

    try:
        with open(pdf_path1, "rb") as file:
            st.download_button(
                label="📄 Download Sample Research Paper",
                data=file,
                file_name="Generative AI Survey.pdf",
                mime="application/pdf"
            )
    except Exception:
        pass

    st.markdown("---")
    st.markdown("🧠 **Powered by**")
    st.markdown("- Groq LLMs (OpenAI GPT OSS)")
    st.markdown("- Hugging Face Embeddings")
    st.markdown("- arXiv + Wikipedia APIs")
<<<<<<< HEAD
    st.markdown("- FAISS DB + Langchain")
=======
    st.markdown("- AutoGen Multi-Agent Framework")
    st.markdown("- FAISS DB")
>>>>>>> 08188d06bd81b0b686b6c6524c7490282ce2f262

    st.markdown("---")
    st.caption("Upload research papers → ask questions → get AI-powered insights")
    st.caption("Developed By Shyam Kumar")


if __name__ == "__main__":
    main()
