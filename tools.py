from langchain_core.tools import tool

@tool
def calculator(operation: str):
    """
    Useful for calculating math expressions.
    Example: '12 * 8' or '2 ** 10'.
    """
    try:
        return f"Result: {eval(operation)}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def manage_tasks(action: str, task_description: str = None):
    """
    Use this tool to add items to the task list.
    Action must be 'add'. task_description is required.
    """
    if action == "add" and task_description:
        return f"SUCCESS: Task '{task_description}' processed."
    return "Error: Invalid action."