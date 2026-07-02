from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from src.graph import build_graph

app = FastAPI(title="ABC Technologies Support Automation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_app = build_graph()


class ChatRequest(BaseModel):
    customer_id: str
    customer_name: str
    query: str


class ApprovalRequest(BaseModel):
    customer_id: str
    decision: str
    notes: str = ""


def _serialize_result(result: dict) -> dict:
    if "__interrupt__" in result:
        data = result["__interrupt__"][0].value
        return {
            "status": "needs_approval",
            "customer": data["customer"],
            "query": data["query"],
            "draft_response": data["draft_response"],
        }
    return {
        "status": "complete",
        "intent": result.get("intent"),
        "final_response": result.get("final_response"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.customer_id}}
    result = graph_app.invoke(
        {"customer_id": req.customer_id, "customer_name": req.customer_name, "query": req.query},
        config=config,
    )
    return _serialize_result(result)


@app.post("/approve")
def approve(req: ApprovalRequest):
    config = {"configurable": {"thread_id": req.customer_id}}
    result = graph_app.invoke(
        Command(resume={"decision": req.decision, "notes": req.notes}),
        config=config,
    )
    return _serialize_result(result)
