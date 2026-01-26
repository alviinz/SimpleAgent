from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 'add_messages' faz com que novas mensagens sejam anexadas ao histórico em vez de sobrescritas
    messages: Annotated[list, add_messages]
    # Lista para o desafio da "Lista de Tarefas"
    tasks: list[str]
    # Para o desafio do "Resumo"
    summary: str