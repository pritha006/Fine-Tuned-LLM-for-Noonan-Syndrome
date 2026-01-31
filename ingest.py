import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings

def build_index():
    df = pd.read_csv("llmdataset.csv")

    documents = []
    for _, row in df.iterrows():
        text = " ".join([str(v) for v in row.values])
        documents.append(Document(text=text))

    embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")

    print("✅ Index created successfully")

if __name__ == "__main__":
    build_index()
