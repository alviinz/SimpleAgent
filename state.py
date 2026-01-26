from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

def merge_tasks(left: list[str], right: list[str]) -> list[str]:
    """Reducer: Combine the old list with the new one and remove duplicates."""
    return list(set((left or []) + (right or [])))

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tasks: Annotated[list[str], merge_tasks]