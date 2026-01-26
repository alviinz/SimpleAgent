import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from state import AgentState

load_dotenv()

# setting up the LLM
llm = ChatOpenAI(
    model="Llama-3.3-70B-Instruct",
    api_key=os.getenv("GITHUB_TOKEN"), 
    base_url="https://models.inference.ai.azure.com"
)

# chatbot node
def chatbot(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# graph construction
workflow = StateGraph(AgentState)

workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

graph = workflow.compile()