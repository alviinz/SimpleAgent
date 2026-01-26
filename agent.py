import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from state import AgentState
from tools import calculator, manage_tasks

load_dotenv()

# setting up the LLM
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
llm_with_tools = llm.bind_tools([calculator, manage_tasks])

# 2. chatbot node
def chatbot(state: AgentState):
    """Chatbot node that processes messages using the LLM."""

    current_tasks = state.get("tasks", [])
    system_msg = SystemMessage(content=f"You are a helpful assistant. Current User Tasks: {current_tasks}")
    messages = [system_msg] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 3. node memory
def task_memory(state: AgentState):
    """Memory node that extracts tasks from tool calls."""
    recent_messages = state["messages"][-3:] 
    new_tasks = []
    
    for msg in recent_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                args = call.get("args", {})
                task_desc = args.get("task_description")
                if task_desc:
                    new_tasks.append(task_desc)
    
    return {"tasks": new_tasks}

# 4. nodes construct
workflow = StateGraph(AgentState)

# adding nodes
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode([calculator, manage_tasks]))
workflow.add_node("task_memory", task_memory)
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    tools_condition,
)
workflow.add_edge("tools", "task_memory")
workflow.add_edge("task_memory", "chatbot")

graph = workflow.compile()