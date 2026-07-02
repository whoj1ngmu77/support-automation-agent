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
        "using only the documentation above. Do not invent policy details."
    )
    response = llm.invoke(prompt)
    return response.content
