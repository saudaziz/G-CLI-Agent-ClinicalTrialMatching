import os
import json
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

# --- 1. LLM Factory (Multi-Model Support) ---

from llama_index.core import Settings

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    
    if provider == "anthropic":
        Settings.embed_model = "local"
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    elif provider == "gemini":
        Settings.embed_model = "local"
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif provider == "ollama":
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "gemma4:latest")
        
        Settings.llm = Ollama(model=model, base_url=base_url, request_timeout=300.0)
        # Use a local embedding model that runs on your CPU/GPU directly
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        return ChatOllama(model=model, base_url=base_url, temperature=0)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

llm = get_llm()

# Load protocol
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()

# Summary Index for high-level criteria
summary_index = SummaryIndex.from_documents(documents)
# Vector Index for specific lab thresholds
vector_index = VectorStoreIndex.from_documents(documents)

# Create tools for the RAG index
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
vector_query_engine = vector_index.as_query_engine()

rag_tools = [
    QueryEngineTool(
        query_engine=summary_query_engine,
        metadata=ToolMetadata(
            name="summary_index_tool",
            description="Use this for high-level inclusion criteria and general trial protocol summary."
        )
    ),
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="vector_index_tool",
            description="Use this for specific lab thresholds, exclusion criteria, and precise numeric data."
        )
    )
]

# --- 2. MCP Server Tool Simulation ---
# In a real environment, this would use mcp-python-sdk to call the live server.
# Here we mock it as a standard LangChain tool for the Researcher Agent.

@tool
def get_patient_data(patient_id: str) -> str:
    """Fetch lab results and doctor notes for a specific patient ID from the legacy SQL database."""
    # Mock data matching the MCP server's PATIENT_DATA
    mock_data = {
        "P001": {
            "name": "John Doe", "age": 45,
            "lab_results": {"HbA1c": "7.2%", "ALT": "45 U/L", "AST": "38 U/L", "eGFR": "85 mL/min/1.73m2"},
            "doctor_notes": "Patient has history of Type 2 Diabetes. Shows interest in clinical trials. No cardiovascular disease. Stable."
        },
        "P002": {
            "name": "Jane Smith", "age": 58,
            "lab_results": {"HbA1c": "8.5%", "ALT": "110 U/L", "AST": "95 U/L", "eGFR": "55 mL/min/1.73m2"},
            "doctor_notes": "Elevated liver enzymes. NAFLD possible. Chronic kidney disease Stage 3a."
        }
    }
    data = mock_data.get(patient_id, f"No data found for patient ID: {patient_id}")
    return json.dumps(data)

# --- 3. Multi-Agent Flow (LangGraph) ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    patient_id: str
    patient_data: str
    trial_rules: str
    match_status: str
    justification: str

# Researcher Agent Node
def researcher_agent(state: AgentState):
    print("--- RESEARCHER AGENT ---")
    patient_id = state['patient_id']
    
    # 1. Get patient data (using tool)
    patient_info = get_patient_data.invoke({"patient_id": patient_id})
    
    # 2. Query RAG for trial rules
    # Simplified call to RAG for this MVP
    summary = summary_query_engine.query("What are the inclusion criteria for this trial?")
    labs = vector_query_engine.query("What are the specific lab thresholds (ALT, AST, eGFR) and exclusion criteria?")
    
    trial_info = f"Summary: {summary}\nLabs: {labs}"
    
    return {
        "patient_data": patient_info,
        "trial_rules": trial_info,
        "messages": [AIMessage(content=f"Gathered data for patient {patient_id} and trial rules.")]
    }

# Orchestrator Agent Node
def orchestrator_agent(state: AgentState):
    print("--- ORCHESTRATOR ---")
    patient_data = state['patient_data']
    trial_rules = state['trial_rules']
    
    prompt = f"""
    Compare the patient data with the trial rules and decide if the patient matches.
    
    Patient Data:
    {patient_data}
    
    Trial Rules:
    {trial_rules}
    
    Provide a decision: MATCH or NO MATCH, followed by a brief reason.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    match_status = "MATCH" if "MATCH" in content.upper() and "NO MATCH" not in content.upper() else "NO MATCH"
    
    return {
        "match_status": match_status,
        "justification": content,
        "messages": [AIMessage(content=f"Match Status: {match_status}")]
    }

# Executor Agent Node
def executor_agent(state: AgentState):
    print("--- EXECUTOR ---")
    patient_id = state['patient_id']
    match_status = state['match_status']
    justification = state['justification']
    
    report = f"""
# Clinical Trial Match Justification Report
**Patient ID:** {patient_id}
**Match Status:** {match_status}

## Detailed Analysis
{justification}

---
Generated by Executor Agent
    """
    
    return {
        "justification": report,
        "messages": [AIMessage(content="Generated final report.")]
    }

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_agent)
workflow.add_node("orchestrator", orchestrator_agent)
workflow.add_node("executor", executor_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "orchestrator")
workflow.add_edge("orchestrator", "executor")
workflow.add_edge("executor", END)

app = workflow.compile()

# --- Execution ---
if __name__ == "__main__":
    initial_state = {
        "patient_id": "P001",
        "messages": [HumanMessage(content="Evaluate patient P001 for clinical trial matching.")],
        "patient_data": "",
        "trial_rules": "",
        "match_status": "",
        "justification": ""
    }
    
    final_output = app.invoke(initial_state)
    print("\nFINAL REPORT:\n")
    print(final_output['justification'])
