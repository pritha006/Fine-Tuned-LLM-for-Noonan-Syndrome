from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.huggingface import HuggingFaceLLM

def load_query_engine():

    llm = HuggingFaceLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={
            "temperature": 0.3,
            "top_p": 0.9
        },
        device_map="auto"
    )

    Settings.llm = llm

    storage_context = StorageContext.from_defaults(
        persist_dir="storage"
    )

    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(
        similarity_top_k=2
    )

    return query_engine
