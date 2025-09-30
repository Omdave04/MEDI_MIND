# streamlit_app.py  -- Chat-only fixed (no experimental_rerun)
import os
import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="MediMind - Streamlit", layout="centered")
st.title("✨ MediMind — Mental Health AI Assistant (Streamlit)")
st.markdown("This app uses **PDF RAG + Gemini LLM** to answer your mental health questions.")

# API KEY
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if not GOOGLE_API_KEY:
    st.warning("⚠️ Google API key not set. Add GOOGLE_API_KEY in Streamlit secrets or env vars.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load PDF -> FAISS (cached)
@st.cache_resource
def load_vectorstore():
    try:
        pdf_path = "mental_health_Document.pdf"
        if not os.path.exists(pdf_path):
            return None, 0
        pdf_reader = PdfReader(pdf_path)
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
        # return None and 0 if anything fails
        return None, 0

vectorstore, chunks_count = load_vectorstore()
if chunks_count:
    st.info(f"📄 Loaded {chunks_count} text chunks from PDF.")
else:
    st.info("⚠️ PDF not loaded. Place 'mental_health_Document.pdf' in the app folder or upload via sidebar.")

# Init LLM
llm = None
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
except Exception as e:
    st.error("❌ Gemini LLM not initialized. Check your API key and installed packages.")
    st.caption(str(e))

# Helper
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

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of tuples: (speaker, text)
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# Chat UI
st.subheader("💬 Chat with Document (RAG + Gemini)")

# Text input bound to session_state
user_msg = st.text_input("Your question:", key="chat_input")

# Buttons
col1, col2 = st.columns([1, 0.25])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Chat")

# Clear chat action (no rerun)
if clear:
    st.session_state.history = []
    st.session_state.chat_input = ""
    st.success("Chat cleared.")

# Send action (safe update, no st.experimental_rerun)
if send:
    msg = user_msg.strip()
    if not msg:
        st.warning("Please type a question before sending.")
    else:
        # append user message
        st.session_state.history.append(("You", msg))

        # get reply (synchronous call)
        with st.spinner("Thinking..."):
            reply = get_pdf_response(msg, k=3)

        st.session_state.history.append(("MediMind", reply))

        # clear the input box by updating session state
        st.session_state.chat_input = ""

# Display history (most recent at bottom)
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**{speaker}:** {text}")

# Sidebar for uploading PDF (simple)
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_pdf = st.file_uploader("Upload a new PDF", type=["pdf"])
    if uploaded_pdf:
        with open("mental_health_Document.pdf", "wb") as f:
            f.write(uploaded_pdf.read())
        st.success("Uploaded PDF saved as 'mental_health_Document.pdf'. Restart the app to re-index.")

