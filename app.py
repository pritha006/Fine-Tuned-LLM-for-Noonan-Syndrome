import os
import streamlit as st

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------- EMBEDDINGS (LOCAL) ----------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# IMPORTANT: Do NOT set Settings.llm at all
# This avoids OpenAI completely

# ---------------- LOAD DOCUMENTS ----------------
documents = SimpleDirectoryReader(
    input_dir=DATA_DIR,
    required_exts=[".txt"]
).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Noonan Syndrome Assistant",
    layout="centered"
)

st.title("🧬 Noonan Syndrome Medical Assistant")
st.write("Ask questions based on the Noonan Syndrome knowledge base.")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Searching medical knowledge..."):
        response = query_engine.query(query)

    st.subheader("Answer")
    st.write(response.response)

