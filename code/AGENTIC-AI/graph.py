# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents import supervisor_agent, research_agent, code_agent

class AgentState(TypedDict):
    task: str
    result: Optional[str]
    next: Optional[str]

builder = StateGraph(AgentState)

builder.add_node("supervisor_agent", supervisor_agent)
builder.add_node("research_agent", research_agent)
builder.add_node("code_agent", code_agent)

def route_to_agent(state):
    return state.get("next", "research_agent")

builder.set_entry_point("supervisor_agent")
builder.add_conditional_edges(
    "supervisor_agent",
    route_to_agent,
    {
        "research_agent": "research_agent",
        "code_agent": "code_agent"
    }
)

builder.add_edge("research_agent", END)
builder.add_edge("code_agent", END)

graph = builder.compile()
