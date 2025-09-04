<!-- # Agentic AI

Agentic AI refers to autonomous systems powered by LLMs and other tools that plan, reason, and act toward goals. These agents interact with APIs, tools, and environments, adapting their behavior based on feedback and context.

## Key Concepts

- **Agent**: Autonomous entity that perceives, reasons, and acts to achieve objectives.
- **Tool Use**: Interaction with external APIs, databases, or software tools.
- **Planning**: Sequencing actions to accomplish complex goals.
- **Reasoning**: Decision-making based on context, feedback, and available information.
- **Memory**: Storing and recalling information across interactions.
- **Multi-Agent Collaboration**: Multiple agents coordinating or working together.
- **Feedback Loop**: Adapting actions based on results or user feedback.

## Popular Agentic AI Frameworks & Solutions

| Framework/Provider           | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| **LangChain Agents**         | Build autonomous agents with LLMs, tool use, planning, and memory. Integrates with multiple LLM providers. |
| **AutoGPT**                  | Open-source agentic AI chaining LLM calls to achieve goals autonomously. Supports plugins and multi-provider LLMs. |
| **BabyAGI**                  | Lightweight agentic AI for task management and autonomous workflows.        |
| **CrewAI**                   | Multi-agent framework for collaborative agentic AI.                        |
| **MetaGPT**                  | Agentic AI for software engineering tasks and multi-agent collaboration.    |
| **OpenAI Function Calling**  | Enables agents to interact with external tools and APIs via LLMs.           |
| **Microsoft Semantic Kernel**| Framework for agentic AI with planning, memory, and tool integration.       |
| **LlamaIndex Agents**        | Agentic capabilities for connecting LLMs to tools and data sources.         |


## Example: LangChain Agent with Multiple Tools (Python)

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Define two tools
def get_weather(location):
    return f"The weather in {location} is sunny."

def get_time(location):
    return f"The current time in {location} is 3:00 PM."

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Provides weather information for a given location."
)

time_tool = Tool(
    name="Time",
    func=get_time,
    description="Provides current time for a given location."
)

llm = OpenAI()
agent = initialize_agent([weather_tool, time_tool], llm, agent_type="zero-shot-react-description")

# Run the agent with a query that could use both tools
response = agent.run("What's the weather and time in London?")
print(response)
```
# Output
The weather in London is sunny. The current time in London is 3:00 PM. -->
