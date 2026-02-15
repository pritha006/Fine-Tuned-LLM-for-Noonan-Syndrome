from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import pandas as pd
import os

# ---------- Embedding ----------
Settings.embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# ---------- Load CSV ----------
df = pd.read_csv("llmdataset.csv")

# ---------- Clean column names ----------
df.columns = df.columns.str.strip()

print("✅ Cleaned Columns:", df.columns.tolist())

documents = []

for _, row in df.iterrows():
    text = f"""
    Question: {row['Question']}
    Correct Medical Answer: {row['Golden Answer']}
    """
    documents.append(Document(text=text))

# ---------- Create index ----------
index = VectorStoreIndex.from_documents(documents)

# ---------- Persist ----------
index.storage_context.persist(persist_dir="storage")

print("✅ Noonan Syndrome index created successfully")


