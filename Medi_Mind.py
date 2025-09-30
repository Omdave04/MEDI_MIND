# streamlit_app.py
import os
import joblib
import pandas as pd
import streamlit as st
from typing import List

# ---- LLM / RAG imports (keep as in your original environment) ----
# Note: keep your installed packages consistent with original codebase
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
# If you used pypdf or PyPDF2 originally, adjust accordingly:
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------- App config ----------
st.set_page_config(page_title="MediMind - Streamlit", layout="centered")

st.title("✨ MediMind — Mental Health AI Assistant (Streamlit)")

# ---- Help text ----
st.markdown(
    "This Streamlit app wraps your existing agent:\n"
    "- RAG (PDF -> FAISS) + Gemini LLM for QA\n"
    "- ML model for treatment prediction\n\n"
    "Make sure model & data files are in the same folder (or upload them below)."
)

# ---- API key (Gemini / Google) ----
# Prefer st.secrets for deployment. Fallback to environment variables.
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if not GOOGLE_API_KEY:
    st.warning("Google API key not set. Set GOOGLE_API_KEY in Streamlit secrets or env vars for LLM to work.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---- Initialize LLM (wrapped in try) ----
llm = None
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
except Exception as e:
    st.info("Warning: could not initialize Gemini LLM locally. LLM-dependent features will fail until key & package are available.")
    st.caption(str(e))

# ---- Load models and resources (from disk) ----
@st.cache_resource
def load_artifacts():
    results = {}
    # Load ML model + encoders + train feature names
    try:
        results["ml_model"] = joblib.load("mental_health_model.pkl")
        results["le_dict"] = joblib.load("label_encoders.pkl")
        results["train_features"] = joblib.load("feature_names.pkl")
    except Exception as e:
        results["ml_model"] = None
        results["le_dict"] = {}
        results["train_features"] = []
        results["load_err"] = f"Model files not found or failed to load: {e}"

    # Load PDF and build vectorstore (FAISS) using HuggingFace embeddings
    try:
        pdf_path = "mental_health_Document.pdf"
        if os.path.exists(pdf_path):
            pdf_reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text() or ""
                pdf_text += text + "\n"
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(pdf_text)
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_texts(chunks, embedding_model)
            results["vectorstore"] = vectorstore
            results["pdf_chunks_count"] = len(chunks)
        else:
            results["vectorstore"] = None
            results["pdf_chunks_count"] = 0
    except Exception as e:
        results["vectorstore"] = None
        results["pdf_chunks_count"] = 0
        results["pdf_err"] = str(e)

    return results

artifacts = load_artifacts()
ml_model = artifacts.get("ml_model")
le_dict = artifacts.get("le_dict", {})
train_features: List[str] = artifacts.get("train_features", [])

# show status
col1, col2 = st.columns(2)
with col1:
    st.metric("ML Model", "Loaded" if ml_model is not None else "Not loaded")
with col2:
    st.metric("PDF chunks", artifacts.get("pdf_chunks_count", 0))

# ---- helper functions (adapted from your code) ----
def predict_treatment(sample: dict) -> str:
    """Run ML model on structured user input and predict treatment need."""
    if ml_model is None:
        return "ML model not available."

    sample_df = pd.DataFrame([sample])
    # encode categorical features
    for col in sample_df.columns:
        if col in le_dict:
            le = le_dict[col]
            # safe transform
            try:
                if sample_df[col].iloc[0] not in le.classes_:
                    sample_df[col] = le.transform([le.classes_[0]])
                else:
                    sample_df[col] = le.transform(sample_df[col])
            except Exception:
                sample_df[col] = le.transform([le.classes_[0]])
    # ensure all train features exist
    for col in train_features:
        if col not in sample_df.columns:
            sample_df[col] = 0
    sample_df = sample_df[train_features]
    # predict
    prediction = ml_model.predict(sample_df)[0]
    return "🔹 Treatment Needed" if prediction == 1 else "✅ No Treatment Needed"

def get_pdf_response(query: str, k: int = 3) -> str:
    """Run similarity search on FAISS and ask LLM. Returns string."""
    vectorstore = artifacts.get("vectorstore")
    if vectorstore is None:
        return "Document knowledge base not available."
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
You are a mental health assistant. Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer clearly and helpfully.
"""
    if llm is None:
        return "LLM not configured. Install and configure Gemini (Google) to enable this."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"LLM invocation failed: {e}"

# ---- Tabs: Chat & Predict ----
tabs = st.tabs(["💬 Chat with Document (RAG + LLM)", "🔮 Predict Treatment (ML model)"])

# ---------- Chat Tab ----------
with tabs[0]:
    st.subheader("Ask questions about the document / mental-health guidance")
    st.write("This uses the PDF RAG and the Gemini LLM (if available).")
    if artifacts.get("pdf_chunks_count", 0) == 0:
        st.info("No PDF loaded. Upload a PDF in the sidebar or place 'mental_health_Document.pdf' in the app folder.")
    # session history
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (user, bot)

    # input
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
        with st.spinner("Fetching answer..."):
            reply = get_pdf_response(msg, k=3)
        st.session_state.history.append(("MediMind", reply))
        st.experimental_rerun()

    # display history
    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**{speaker}:** {text}")

# ---------- Predict Tab ----------
with tabs[1]:
    st.subheader("Run treatment prediction (from your ML model)")
    if ml_model is None:
        st.error("ML model not loaded. Place 'mental_health_model.pkl' and related files in the app folder.")
    else:
        st.write("Fill the fields (options use your label encoders where available). Leave blank to use defaults.")
        # dynamically render inputs for each train_feature
        feature_inputs = {}
        cols = st.columns(2)
        for i, feat in enumerate(train_features):
            col = cols[i % 2]
            if feat in le_dict:
                opts = [str(x) for x in le_dict[feat].classes_.tolist()]
                val = col.selectbox(feat, options=[""] + opts, index=0, key=f"feat_{i}")
                feature_inputs[feat] = val if val != "" else ""
            else:
                val = col.text_input(feat, key=f"feat_txt_{i}")
                feature_inputs[feat] = val

        if st.button("Run Prediction"):
            # sanitize values -> default "0" for blanks to match your old logic
            sample = {f: ("" if v is None else str(v)).strip() for f, v in feature_inputs.items()}
            # use "0" for blanks (like original)
            sample = {k: (v if v != "" else "0") for k, v in sample.items()}
            result = predict_treatment(sample)
            st.info(result)

        if st.button("Fill Example"):
            # example values (you can adjust)
            EXAMPLE_SAMPLE = {
                "Gender": "Female",
                "family_history": "Yes",
                "Age": "28",
                "self_employed": "Yes",
                "work_interfere": "Always",
                "benefits": "No",
                "care_options": "No",
                "anonymity": "No",
                "leave": "Very difficult",
                "Social_Weakness": "Yes",
            }
            for i, feat in enumerate(train_features):
                if feat in EXAMPLE_SAMPLE:
                    st.session_state[f"feat_{i}"] = EXAMPLE_SAMPLE[feat] if feat in le_dict else EXAMPLE_SAMPLE[feat]
            st.experimental_rerun()

# ---- Sidebar: Upload / files / info ----
with st.sidebar:
    st.header("Files & Settings")
    st.write("You can upload a PDF (will replace the local PDF for the session).")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf is not None:
        # save uploaded pdf to disk and rebuild vectorstore (quick simple approach)
        pdf_out = "uploaded_document.pdf"
        with open(pdf_out, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success("Saved uploaded PDF as uploaded_document.pdf. Restart the app to re-index (or reload).")
        st.info("For production: implement re-index function to re-create FAISS instantly.")
    st.markdown("---")
    st.markdown("### Deployment notes")
    st.write(
        "- Set `GOOGLE_API_KEY` in Streamlit secrets (or environment) to enable Gemini LLM.\n"
        "- Make sure the model files (`mental_health_model.pkl`, `label_encoders.pkl`, `feature_names.pkl`) "
        "are present in the app folder on the server."
    )
    st.markdown("---")
    st.write("MediMind - AI Powered ChatBot")

