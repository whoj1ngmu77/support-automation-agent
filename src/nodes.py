from src.state import SupportState
from src.llm import classify_intent_llm, generate_agent_response
from src.rag import KnowledgeBase

_kb = KnowledgeBase()


def classify_intent_node(state: SupportState) -> dict:
    intent = classify_intent_llm(state["query"])
    return {"intent": intent}


def _department_agent(department: str, state: SupportState) -> dict:
    context = _kb.retrieve(state["query"], department=department)
    response = generate_agent_response(department, state["query"], context)
    return {
        "retrieved_context": context,
        "draft_response": response,
        "final_response": response,   # placeholder until Phase 7 (supervisor)
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
    # Placeholder — replaced with real SQLite-backed recall in Phase 5.
    return {"final_response": f"[Memory placeholder] Got your query: {state['query']}"}
