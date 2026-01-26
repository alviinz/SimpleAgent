import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from state import AgentState
from tools import calculator, manage_tasks, save_summary
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# setting up the LLM
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
llm_with_tools = llm.bind_tools([calculator, manage_tasks, save_summary])

# 2. chatbot node
def chatbot(state: AgentState):
    """Chatbot node that processes messages using the LLM."""

    current_tasks = state.get("tasks", [])
    current_summary = state.get("summary", "None summary yet.")

    system_msg = SystemMessage(content=(
    f"You are a helpful assistant. "
    f"Current Summary: {current_summary}. "
    f"Current User Tasks: {current_tasks}. "
    f"IMPORTANT: Do not use LaTeX formatting (like \\( \\) or \\times). "
    f"Always output math in plain text (e.g., '2 * 5 = 10')."
    f"Do not use decimal points for whole numbers."
))
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# 3. node memory
def memory_manager(state: AgentState):
    """Memory node that extracts tasks from tool calls."""
    recent_messages = state["messages"][-2:]
    updates = {}
    new_tasks = []
    
    for msg in recent_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                args = call.get("args", {})
                tool_name = call.get("name")
                
                if tool_name == "manage_tasks":
                    if args.get("action") == "add" and args.get("task_description"):
                        new_tasks.append(args["task_description"])
                
                elif tool_name == "save_summary":
                    updates["summary"] = args.get("summary_text")
    
    if new_tasks:
        updates["tasks"] = new_tasks
        
    return updates

# graphs construction
workflow = StateGraph(AgentState)

workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode([calculator, manage_tasks, save_summary]))
workflow.add_node("memory_manager", memory_manager)

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "memory_manager")
workflow.add_edge("memory_manager", "chatbot")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("\n🤖 Bot iniciado! (Digite 'sair' para fechar)")
    
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        try:
            user_input = input("\nVocê: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("\nEncerrando...")
                break
            
            events = graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            for event in events:
                for node_name, values in event.items():
                    if values and "messages" in values:
                        last_msg = values["messages"][-1]
                        print(f"\n--- {node_name} ---\n")
                        print(last_msg.content)
                    
                    elif values and "summary" in values:
                        print(f"\n--- {node_name} ---\n")
                        print(f"[Sistema] Resumo atualizado.")
                        
        except Exception as e:
            print(f"Erro no loop: {repr(e)}")