# llm_groq.py

# Import ChatOpenAI from langchain_openai package
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Set your Groq API key (consider using environment variables for security)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the ChatOpenAI LLM with Groq API settings
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",      # Groq API endpoint
    api_key=GROQ_API_KEY,                           # API key for authentication
    model="llama-3.1-8b-instant",                   # Specify the model to use
    # temperature=0.7,                              # Optional: set temperature for response randomness
    streaming=True,                                 # Enable streaming responses
)
