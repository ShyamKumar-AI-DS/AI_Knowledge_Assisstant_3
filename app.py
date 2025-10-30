# ai_knowledge_assistant_app.py
"""
AI Knowledge Assistant (RAG + Groq-native Summarizer/Explainer)
- FAISS in-memory (session_state)
- PDF upload + external fetch (arXiv + Wikipedia)
- No AutoGen, no OpenAI dependency
"""

import os
import tempfile
import streamlit as st
import arxiv
import wikipedia
from dotenv import load_dotenv

# LangChain
from langchain_groq import ChatGroq     
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# -------------------------
# Load LLM (Groq-based)
# -------------------------
load_dotenv()

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=1500,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# -------------------------
# PDF ingestion → FAISS in memory
# -------------------------
def ingest_pdf_to_faiss(uploaded_file, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(raw_docs)
    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if 'faiss_db' in st.session_state:
        st.session_state['faiss_db'].add_documents(docs)
    else:
        st.session_state['faiss_db'] = FAISS.from_documents(docs, hf_embeddings)

    os.unlink(temp_path)
    return st.session_state['faiss_db']

# -------------------------
# External knowledge
# -------------------------
def fetch_arxiv_papers(query, max_results=3):
    results = []
    search = arxiv.Search(query=query, max_results=max_results)
    for r in search.results():
        results.append({
            "title": r.title,
            "summary": r.summary,
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id
        })
    return results

def fetch_wikipedia_summary(topic, sentences=3):
    try:
        page = wikipedia.page(topic)
        summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=True, redirect=True)
        return {"title": page.title, "summary": summary, "url": page.url}
    except Exception as e:
        return {"title": topic, "summary": f"Wikipedia fetch error: {e}", "url": ""}

def add_external_results_to_faiss(external_texts, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    docs = [Document(page_content=item["text"], metadata=item.get("meta", {})) for item in external_texts]

    if 'faiss_db' in st.session_state:
        st.session_state['faiss_db'].add_documents(docs)
    else:
        st.session_state['faiss_db'] = FAISS.from_documents(docs, hf_embeddings)
    return st.session_state['faiss_db']

# -------------------------
# Groq-native Summarizer & Explainer
# -------------------------
def summarize_with_groq(docs_text):
    prompt = PromptTemplate.from_template(
        "Summarize the following documents into 5-6 concise bullet points:\n\n{docs}"
    )
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(docs=docs_text)

def explain_with_groq(docs_text, question):
    prompt = PromptTemplate.from_template(
        "Explain the following context to a beginner, step by step, and then answer the question. "
        "End with a one-sentence summary.\n\nContext:\n{docs}\n\nQuestion: {question}"
    )
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(docs=docs_text, question=question)

# -------------------------
# Orchestration (replaces AutoGen router)
# -------------------------
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
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Assuming you have a function to get your LLM instance
# from your_llm_module import get_llm 

def build_retrieval_qa(llm, k=3):
    if 'faiss_db' not in st.session_state:
        raise ValueError("No FAISS index found. Please upload PDF documents first.")
    db = st.session_state['faiss_db']
    retriever = db.as_retriever(search_kwargs={"k": k})
    prompt_template = """Answer the user's question based only on the following context:

    <context>
    {context}
    </context>

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain, retriever

# def build_retrieval_qa(embedding_model="sentence-transformers/all-MiniLM-L6-v2", k=3):
#     if 'faiss_db' not in st.session_state:
#         raise ValueError("No FAISS index found. Upload PDFs first.")
#     db = st.session_state['faiss_db']
#     retriever = db.as_retriever(search_kwargs={"k": k})
#     qa_chain = create_retrieval_chain.from_chain_type(llm=get_llm(), retriever=retriever, return_source_documents=True)
#     return qa_chain, retriever

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
    st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
    inject_custom_css()
    st.title("🤖 AI Knowledge Assistant (RAG + AutoGen Agents)")

    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            ingest_pdf_to_faiss(f)
        st.success("✅ Papers ingested successfully!")

    question = st.text_input("Enter your question")
    with st.expander("⚙️ External Knowledge Settings"):
        do_external = st.checkbox("Fetch external results (arXiv + Wikipedia)", value=True)
        arxiv_max = st.slider("Number of arXiv results", 1, 10, 3)

    if st.button("🚀 Run Agents") and question.strip():
        with st.spinner("Running AutoGen agents..."):
            qa_chain, retriever = build_retrieval_qa()
            if do_external:
                st.info("Fetching external knowledge...")
                arx = fetch_arxiv_papers(question, max_results=arxiv_max)
                wiki = fetch_wikipedia_summary(question)
                st.session_state.external = {"arxiv": arx, "wiki": wiki}
            # call the autogen-based router
            result = run_agents(question, retriever, qa_chain)

        st.subheader("📌 Answer")
        st.markdown(f"<div class='result-box'>{result['concise_answer']}</div>", unsafe_allow_html=True)

        st.subheader("📝 Explanation")
        st.markdown(f"<div class='result-box'>{result['explanation']}</div>", unsafe_allow_html=True)

        st.subheader("🔍 Summary")
        st.markdown(f"<div class='result-box'>{result['summary']}</div>", unsafe_allow_html=True)

        st.subheader("📚 Sources")
        for d in result["source_documents"][:5]:
            st.markdown(f"<div class='result-box'>{getattr(d,'page_content','')[:300]}...</div>", unsafe_allow_html=True)

        if "external" in st.session_state:
            for a in st.session_state["external"].get("arxiv", []):
                st.markdown(f"- [{a['title']}]({a['pdf_url']})")
            wiki = st.session_state["external"].get("wiki", {})
            if wiki:
                st.markdown(f"- **Wikipedia:** [{wiki['title']}]({wiki['url']}) — {wiki['summary']}")
with st.sidebar:
    st.subheader("📘 AI Knowledge Assistant")
    st.info("""
    This AI agent combines:
    - **Generative Text + RAG + PDF Ingestion** for document understanding
    - **arXiv + Wikipedia** for external context
    - **AutoGen Multi-Agent Orchestration** for summarization and explanation
    - **FAISS (Facebook AI Similarity Search)**
    
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
    st.markdown("- AutoGen Multi-Agent Framework")
    st.markdown("- FAISS")

    st.markdown("---")
    st.caption("Upload research papers → ask questions → get AI-powered insights")
    st.caption("Developed By Shyam Kumar")
if __name__ == "__main__":
    main()
