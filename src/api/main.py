from src.config import *
from src.api.agent.agent import AgentSquad
from src.api.agent.session_store import session_store
from src.logging import logger
from fastapi import FastAPI, Header, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from fastapi.middleware.cors import CORSMiddleware
import uuid
import asyncio
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[FRONTEND_URL],
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"], 
    allow_headers=["*"],  
)
@app.get("/health")
async def health():
    return {"status": "ok"}

class InvokeRequest(BaseModel):
    content: str

@app.post("/invoke")
async def invoke(request: InvokeRequest, x_session_id: str | None = Header(default=None)):
    session_id = x_session_id or str(uuid.uuid4())
    executor = await session_store.get_or_create(session_id)

    return StreamingResponse(executor.event_generator(request.content, session_id), media_type="text/event-stream")