from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

VALID_INTENTS = {"Sales", "Technical", "Billing", "Account", "Memory"}


def classify_intent_llm(query: str) -> str:
    prompt = (
        "Classify the customer query into exactly one category: "
        "Sales, Technical, Billing, Account, or Memory. "
        "Use Memory only if the customer is asking about a PAST interaction or issue, "
        "not making a new request. Reply with only the category word, nothing else.\n\n"
        f"Query: {query}"
    )
    response = llm.invoke(prompt)
    label = response.content.strip().split()[0]

    if label not in VALID_INTENTS:
        return "Technical"
    return label


def generate_agent_response(department: str, query: str, context: list[str]) -> str:
    context_text = "\n\n".join(context) if context else "No specific documentation found."
    prompt = (
        f"You are the {department} Support agent for ABC Technologies.\n"
        f"Customer query: {query}\n\n"
        f"Relevant documentation:\n{context_text}\n\n"
        "Write a concise, helpful, professional response to the customer, "
        "using only the documentation above. Do not invent policy details. "
        "Do not include a greeting like 'Hello' or 'Dear Customer' — start directly "
        "with the substance of the response, since a greeting is added separately."
    )
    response = llm.invoke(prompt)
    return response.content


def answer_from_memory(query: str, history_text: str) -> str:
    if not history_text.strip():
        return "I don't see any previous interactions on file for you yet."
    prompt = (
        f"Conversation history so far:\n{history_text}\n\n"
        f"Customer now asks: {query}\n"
        "Answer using only the conversation history above."
    )
    response = llm.invoke(prompt)
    return response.content


def supervisor_review(draft: str) -> str:
    prompt = (
        "You are a Supervisor agent reviewing a customer support response before it's sent. "
        "Improve clarity, correctness, and tone if needed. Do NOT include any greeting "
        "(no 'Hello', 'Dear Customer', 'Hi') since one is added separately. Start directly "
        "with the substance of the response. Return only the improved response, nothing else.\n\n"
        f"Draft: {draft}"
    )
    response = llm.invoke(prompt)
    return response.content
