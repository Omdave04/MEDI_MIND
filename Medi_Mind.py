# streamlit_app.py
import os
import streamlit as st

import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------- APP CONFIG -----------------
st.set_page_config(page_title="MediMind - Streamlit", layout="centered")
st.title("✨ MediMind — Mental Health AI Assistant (Streamlit)")
st.markdown("This app uses **PDF RAG + Gemini LLM** to answer your mental health questions.")

# ----------------- API KEY -----------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if not GOOGLE_API_KEY:
    st.warning("⚠️ Google API key not set. Add GOOGLE_API_KEY in Streamlit secrets or env vars.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ----------------- LOAD PDF + VECTORSTORE -----------------
@st.cache_resource
def load_vectorstore():
    try:
        pdf_reader = PdfReader("mental_health_Document.pdf")
        pdf_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text() or ""
            pdf_text += text + "\n"

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(pdf_text)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embedding_model)
        return vectorstore, len(chunks)
    except Exception as e:
        return None, 0

vectorstore, chunks_count = load_vectorstore()
st.info(f"📄 Loaded {chunks_count} text chunks from PDF." if chunks_count else "⚠️ PDF not loaded.")

# ----------------- INIT LLM -----------------
llm = None
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.2)
except Exception as e:
    st.error("❌ Gemini LLM not initialized. Check your API key and packages.")
    st.caption(str(e))

# ----------------- HELPER -----------------
def get_pdf_response(query: str, k: int = 3) -> str:
    if not vectorstore:
        return "❌ Document knowledge base not available."
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
You are a mental health assistant. Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer clearly and helpfully.
"""
    if not llm:
        return "❌ LLM not configured. Please set GOOGLE_API_KEY."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"❌ LLM invocation failed: {e}"

# ----------------- CHAT UI -----------------
st.subheader("💬 Chat with Document (RAG + Gemini)")
if "history" not in st.session_state:
    st.session_state.history = []

msg = st.text_input("Your question:", key="chat_input")
col1, col2 = st.columns([1, 0.2])
with col1:
    send = st.button("Send", key="send_chat")
with col2:
    clear = st.button("Clear", key="clear_chat")

if clear:
    st.session_state.history = []
    st.experimental_rerun()

if send and msg:
    st.session_state.history.append(("You", msg))
    with st.spinner("Thinking..."):
        reply = get_pdf_response(msg, k=3)
    st.session_state.history.append(("MediMind", reply))
    st.experimental_rerun()

# display chat
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**{speaker}:** {text}")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_pdf = st.file_uploader("Upload a new PDF", type=["pdf"])
    if uploaded_pdf:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_pdf.read())
        st.success("Uploaded PDF saved as `uploaded_document.pdf`. Restart app to re-index.")
