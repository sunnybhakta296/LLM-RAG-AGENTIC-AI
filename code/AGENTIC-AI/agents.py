# agents.py
from llm_groq import llm

def research_agent(state):
    prompt = f"Research the topic: {state['task']}"
    response = llm.invoke(prompt)
    return {"result": response.content}

def code_agent(state):
    prompt = f"Write Python code for: {state['task']}"
    response = llm.invoke(prompt)
    return {"result": response.content}

def supervisor_agent(state):
    task = state["task"].lower()
    if "code" in task:
        return {"next": "code_agent"}
    return {"next": "research_agent"}
