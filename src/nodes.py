from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt

from src.state import SupportState
from src.llm import classify_intent_llm, generate_agent_response, answer_from_memory
from src.rag import KnowledgeBase

_kb = KnowledgeBase()

HIGH_RISK_KEYWORDS = [
    "refund", "cancel my subscription", "cancel subscription", "close my account",
    "account closure", "compensation", "escalate to management", "speak to a manager",
]


def classify_intent_node(state: SupportState) -> dict:
    intent = classify_intent_llm(state["query"])
    return {"intent": intent}


def _is_high_risk(query: str, intent: str) -> bool:
    q = query.lower()
    if intent == "Billing" and ("refund" in q or "cancel" in q):
        return True
    return any(kw in q for kw in HIGH_RISK_KEYWORDS)


def _department_agent(department: str, state: SupportState) -> dict:
    context = _kb.retrieve(state["query"], department=department)
    response = generate_agent_response(department, state["query"], context)
    requires_approval = _is_high_risk(state["query"], department)
    return {
        "retrieved_context": context,
        "draft_response": response,
        "requires_approval": requires_approval,
        "approval_status": "pending" if requires_approval else "not_required",
    }


def sales_agent_node(state: SupportState) -> dict:
    return _department_agent("Sales", state)


def technical_agent_node(state: SupportState) -> dict:
    return _department_agent("Technical", state)


def billing_agent_node(state: SupportState) -> dict:
    return _department_agent("Billing", state)


def account_agent_node(state: SupportState) -> dict:
    return _department_agent("Account", state)


def memory_recall_node(state: SupportState) -> dict:
    history_text = "\n".join(
        f"{'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in state.get("messages", [])
    )
    answer = answer_from_memory(state["query"], history_text)
    return {
        "final_response": answer,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=answer)],
    }


def human_approval_node(state: SupportState) -> dict:
    decision = interrupt({
        "reason": "High-risk request requires supervisor approval",
        "customer": state["customer_name"],
        "query": state["query"],
        "draft_response": state["draft_response"],
    })
    approved = decision.get("decision") == "approve"
    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_notes": decision.get("notes", ""),
    }


def finalize_response_node(state: SupportState) -> dict:
    if state.get("approval_status") == "rejected":
        final = (
            f"Hi {state['customer_name']}, your request needs further verification "
            f"before we can proceed. Note: {state.get('approval_notes', '')}"
        )
    else:
        final = state["draft_response"]
    return {
        "final_response": final,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=final)],
    }
