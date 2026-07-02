from src.state import SupportState
from src.llm import classify_intent_llm


def classify_intent_node(state: SupportState) -> dict:
    intent = classify_intent_llm(state["query"])
    return {"intent": intent}


def _placeholder(department: str):
    def node(state: SupportState) -> dict:
        return {"final_response": f"[{department} placeholder] Got your query: {state['query']}"}
    return node


sales_agent_node = _placeholder("Sales")
technical_agent_node = _placeholder("Technical")
billing_agent_node = _placeholder("Billing")
account_agent_node = _placeholder("Account")
memory_recall_node = _placeholder("Memory")
