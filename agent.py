import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from state import PrivateState, InputState
from tools import calculator, manage_tasks, save_summary
from rag_tools import search_faq_database, verify_response_quality
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# setting up the LLM
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv("GROQ_API_KEY"),
)
llm_with_tools = llm.bind_tools([calculator, manage_tasks, save_summary])

# chatbot node
def chatbot(state: PrivateState):
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

def router(state: PrivateState):
    """Node that decides the path: FAQ or Common Chat."""
    question = state["messages"][-1].content.lower()
    
    # Quick prompt for the LLM to decide routing
    prompt = f"The user asked: '{question}'. Is this a technical question about products, cart, or promotions? Respond only YES or NO."
    answer = llm.invoke([HumanMessage(content=prompt)]).content.upper()

    if "YES" in answer:
        return "rag_node"
    return "chatbot"

def rag_node(state: PrivateState):
    """
    Completed RAG: Search - Generate Response - Return.
    """
    question = state["messages"][-1].content
    
    retrieved_text = search_faq_database(question)

    isValid = verify_response_quality(question, retrieved_text)
    
    if not isValid or "I didn't find any FAQ information." in retrieved_text:
        error_prompt = (
            f"The user asked: '{question}'.\n"
            f"We searched the database but did NOT find any official information about that.\n"
            f"Instruction: Respond politely to the user informing that you did not find that information in the official documentation. "
            f"Respond in the same language the user used in the question."
        )
        error_message = llm.invoke([HumanMessage(content=error_prompt)])
        return {"messages": [error_message]}
    prompt_rag = (
        f"You're a trusted seller's assistant. "
        f"Answer the user's question using ONLY the following context retrieved from the database.\n"
        f"--- CONTEXT ---\n{retrieved_text}\n----------------\n"
        f"User's question: {question}\n"
        f"Instruction: Respond in a friendly manner and end with the citation '[Source: Official FAQ]'."
    )

    final_response = llm.invoke([HumanMessage(content=prompt_rag)])
    
    return {"messages": [final_response]}

# memory node
def memory_manager(state: PrivateState):
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

def input_node(state: InputState):
    return {
        "messages": [HumanMessage(content=state["question"])]
    }

# graphs construction
workflow = StateGraph(PrivateState)

workflow.add_node("input_node", input_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode([calculator, manage_tasks, save_summary]))
workflow.add_node("memory_manager", memory_manager)

workflow.add_edge(START, "input_node")
workflow.add_conditional_edges(
    "input_node",
    router,
    {               
        "rag_node": "rag_node",
        "chatbot": "chatbot"
    }
)
workflow.add_edge("rag_node", END)
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "memory_manager")
workflow.add_edge("memory_manager", "chatbot")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("\n Bot iniciado! (Digite 'sair' para fechar)")
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        try:
            user_input = input("\nVocê: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                break
            
            print("\nProcessando...", end="\r")
            
            final_state = graph.invoke(
                {"question": user_input}, 
                config=config
            )
            
            print(f"Assistente: {final_state['messages'][-1].content}")
            
        except Exception as e:
            print(f"Erro: {e}")