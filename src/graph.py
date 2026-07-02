from langgraph.graph import StateGraph, START, END

from src.state import SupportState
from src.nodes import (
    classify_intent_node,
    sales_agent_node,
    technical_agent_node,
    billing_agent_node,
    account_agent_node,
    memory_recall_node,
)


def route_by_intent(state: SupportState) -> str:
    return {
        "Sales": "sales_agent",
        "Technical": "technical_agent",
        "Billing": "billing_agent",
        "Account": "account_agent",
        "Memory": "memory_recall",
    }.get(state["intent"], "technical_agent")


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("sales_agent", sales_agent_node)
    graph.add_node("technical_agent", technical_agent_node)
    graph.add_node("billing_agent", billing_agent_node)
    graph.add_node("account_agent", account_agent_node)
    graph.add_node("memory_recall", memory_recall_node)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_recall": "memory_recall",
        },
    )

    for node in ["sales_agent", "technical_agent", "billing_agent", "account_agent", "memory_recall"]:
        graph.add_edge(node, END)

    return graph.compile()
