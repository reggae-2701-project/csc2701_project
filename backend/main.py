from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from rag_architecture.rag import RAG

app = FastAPI()
rag = RAG(
    embedding_model_name='all-MiniLM-L6-v2',
    language_model_name='gemini-2.5-flash-lite'
)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatHistory(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}

@app.post("/chat")
async def chat(history: ChatHistory):
    try:
        latest_user_message = history.messages[-1]["content"]
        response = rag(user_query=latest_user_message)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {e}"}
