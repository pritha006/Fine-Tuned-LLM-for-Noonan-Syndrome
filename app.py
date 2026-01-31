from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def load_query_engine():
    # ✅ MUST be set FIRST
    embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.embed_model = embed_model

    # ✅ LLM
    llm = HuggingFaceLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
        context_window=4096,
        max_new_tokens=512,
        generate_kwargs={
            "temperature": 0.3,
            "top_p": 0.9
        },
        device_map="auto",
    )
    Settings.llm = llm

    # ✅ Load stored index
    storage_context = StorageContext.from_defaults(
        persist_dir="storage"
    )

    index = load_index_from_storage(storage_context)

    return index.as_query_engine(similarity_top_k=3)
