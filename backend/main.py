from fastapi import FastAPI
from pydantic import BaseModel
from query_engine.py import load_query_engine

app = FastAPI(
    title="Noonan Syndrome AI API",
    description="Semantic Medical Assistant Backend",
)

query_engine = load_query_engine()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "Noonan AI Backend Running"}

@app.post("/chat")
def chat(req: QueryRequest):
    response = query_engine.query(req.question)
    return {"answer": str(response)}
