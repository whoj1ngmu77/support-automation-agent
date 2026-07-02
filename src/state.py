from typing import Annotated, List, Optional, TypedDict
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    customer_id: str
    customer_name: str

    messages: Annotated[list, add_messages]

    query: str
    intent: str
    retrieved_context: List[str]

    requires_approval: bool
    approval_status: str
    approval_notes: Optional[str]

    draft_response: str
    final_response: str
