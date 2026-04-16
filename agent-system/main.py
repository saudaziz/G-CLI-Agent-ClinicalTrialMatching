import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from llama_index.core import Settings, SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from langgraph.graph import END, StateGraph

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

class AgentState(TypedDict):
    messages: List[BaseMessage]
    patient_id: str
    patient_data: str
    trial_rules: str
    match_status: str
    justification: str


def _create_ollama() -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    Settings.llm = Ollama(model=model, base_url=base_url, request_timeout=300.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return ChatOllama(model=model, base_url=base_url, temperature=0)


LLM_FACTORIES = {
    "anthropic": lambda: ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0),
    "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0),
    "ollama": _create_ollama,
}


def get_llm() -> Any:
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

    if provider in {"anthropic", "gemini"}:
        Settings.embed_model = "local"

    factory = LLM_FACTORIES.get(provider)
    if factory is None:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return factory()


def load_documents(data_dir: Path) -> List[Any]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    reader = SimpleDirectoryReader(str(data_dir))
    return reader.load_data()


def build_rag_engines(documents: List[Any]) -> Dict[str, Any]:
    summary_index = SummaryIndex.from_documents(documents)
    vector_index = VectorStoreIndex.from_documents(documents)

    return {
        "summary_query_engine": summary_index.as_query_engine(response_mode="tree_summarize"),
        "vector_query_engine": vector_index.as_query_engine(),
    }


def determine_match_status(content: str) -> str:
    normalized = content.strip().upper()
    if "NO MATCH" in normalized:
        return "NO MATCH"
    if "MATCH" in normalized:
        return "MATCH"
    return "UNKNOWN"


@tool
def get_patient_data(patient_id: str) -> str:
    """Fetch lab results and doctor notes for a specific patient ID from the legacy SQL database."""
    mock_data: Dict[str, Dict[str, Any]] = {
        "P001": {
            "name": "John Doe",
            "age": 45,
            "lab_results": {
                "HbA1c": "7.2%",
                "ALT": "45 U/L",
                "AST": "38 U/L",
                "eGFR": "85 mL/min/1.73m2",
            },
            "doctor_notes": (
                "Patient has history of Type 2 Diabetes. Shows interest in clinical trials. "
                "No cardiovascular disease. Stable."
            ),
        },
        "P002": {
            "name": "Jane Smith",
            "age": 58,
            "lab_results": {
                "HbA1c": "8.5%",
                "ALT": "110 U/L",
                "AST": "95 U/L",
                "eGFR": "55 mL/min/1.73m2",
            },
            "doctor_notes": (
                "Elevated liver enzymes. NAFLD possible. Chronic kidney disease Stage 3a."
            ),
        },
    }

    data = mock_data.get(patient_id)
    if not data:
        logger.warning("No patient found for ID %s", patient_id)
        return json.dumps({"error": f"No data found for patient ID: {patient_id}"})

    return json.dumps(data, indent=2)


def create_workflow(
    llm: Any, summary_query_engine: Any, vector_query_engine: Any
) -> StateGraph:
    def researcher_agent(state: AgentState):
        logger.info("Starting researcher agent for patient %s", state["patient_id"])
        patient_id = state["patient_id"]
        patient_info = get_patient_data.invoke({"patient_id": patient_id})

        summary = summary_query_engine.query(
            "What are the inclusion criteria for this trial?"
        )
        labs = vector_query_engine.query(
            "What are the specific lab thresholds (ALT, AST, eGFR) and exclusion criteria?"
        )

        trial_info = f"Summary: {summary}\nLabs: {labs}"

        return {
            "patient_data": patient_info,
            "trial_rules": trial_info,
            "messages": [
                AIMessage(content=f"Gathered data for patient {patient_id} and trial rules.")
            ],
        }

    def orchestrator_agent(state: AgentState):
        logger.info("Orchestrating decision for patient %s", state["patient_id"])
        prompt = (
            "Compare the patient data with the trial rules and decide if the patient matches.\n\n"
            "Patient Data:\n"
            f"{state['patient_data']}\n\n"
            "Trial Rules:\n"
            f"{state['trial_rules']}\n\n"
            "Provide a decision: MATCH or NO MATCH, followed by a brief reason."
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        match_status = determine_match_status(content)

        return {
            "match_status": match_status,
            "justification": content,
            "messages": [AIMessage(content=f"Match Status: {match_status}")],
        }

    def executor_agent(state: AgentState):
        logger.info("Generating report for patient %s", state["patient_id"])
        report = (
            "# Clinical Trial Match Justification Report\n"
            f"**Patient ID:** {state['patient_id']}\n"
            f"**Match Status:** {state['match_status']}\n\n"
            "## Detailed Analysis\n"
            f"{state['justification']}\n\n"
            "---\nGenerated by Executor Agent"
        )

        return {
            "justification": report,
            "messages": [AIMessage(content="Generated final report.")],
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("executor", executor_agent)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "orchestrator")
    workflow.add_edge("orchestrator", "executor")
    workflow.add_edge("executor", END)

    return workflow


def main() -> None:
    llm = get_llm()
    documents = load_documents(DATA_DIR)
    engines = build_rag_engines(documents)
    workflow = create_workflow(
        llm=llm,
        summary_query_engine=engines["summary_query_engine"],
        vector_query_engine=engines["vector_query_engine"],
    )
    app = workflow.compile()

    initial_state: AgentState = {
        "patient_id": "P001",
        "messages": [HumanMessage(content="Evaluate patient P001 for clinical trial matching.")],
        "patient_data": "",
        "trial_rules": "",
        "match_status": "",
        "justification": "",
    }

    final_output = app.invoke(initial_state)
    logger.info("Final report generated")
    print("\nFINAL REPORT:\n")
    print(final_output["justification"])


if __name__ == "__main__":
    main()
