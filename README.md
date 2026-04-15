# Agentic Clinical Trial Matching System

This project implements an MVP for an Agentic Clinical Trial Matching System using the Model Context Protocol (MCP), LlamaIndex, and LangGraph.

## Architecture

1.  **MCP Server (TypeScript)**: Simulates a legacy SQL database containing patient records (lab results and doctor notes).
2.  **RAG Pipeline (LlamaIndex)**: Indexes Clinical Trial Protocol documents using both a Summary Index (high-level criteria) and a Vector Index (specific lab thresholds).
3.  **Multi-Agent Flow (LangGraph)**:
    - **Researcher Agent**: Fetches patient data via MCP and trial rules via RAG.
    - **Orchestrator Agent**: Analyzes the comparison and determines if there is a match.
    - **Executor Agent**: Generates a detailed Match Justification report in Markdown.

## Multi-Model Support

The system is configurable to use different LLM providers via a toggle in the `.env` file:
- **Anthropic**: Claude 3.5 Sonnet.
- **Gemini**: Google's Gemini 1.5 Flash (free tier).
- **Ollama**: Local or network instances (e.g., `gemma4:latest`).

### Local Embedding Strategy
To avoid `501: model does not support embeddings` errors often found in Ollama generative models, this system uses a **Hybrid RAG approach**:
- **Embeddings**: Handled locally on your machine using the `BAAI/bge-small-en-v1.5` model (automatically downloaded via HuggingFace).
- **Inference**: Handled by your remote/local Ollama server.

## Setup

### 1. MCP Server
```bash
cd mcp-server
npm install
npm run build
```

### 2. Agent System
```bash
cd agent-system
pip install -r requirements.txt
```
- Configure your `.env` file (see Configuration below).
- Place your trial protocol PDF/txt in `agent-system/data/`.

## Configuration (`agent-system/.env`)

Ensure your `.env` matches your environment. For the network instance used during development:

```env
# Toggle between: anthropic, gemini, ollama
LLM_PROVIDER=ollama

# API Keys (Required only for cloud providers)
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key

# Ollama Configuration
OLLAMA_BASE_URL=http://192.168.68.190:11434
OLLAMA_MODEL=gemma4:latest
```

**Note:** `OLLAMA_BASE_URL` must be the **root URL** (e.g., `http://IP:11434`), not an API endpoint like `/api/tags`.

## Running the MVP

1.  **Start the MCP Server** (in one terminal):
    ```bash
    cd mcp-server
    npm start
    ```
2.  **Run the Multi-Agent Workflow**:
    ```bash
    cd agent-system
    python main.py
    ```

## Troubleshooting

- **Connection Error**: Ensure your Ollama server is running and accessible via the specified IP.
- **Model Not Found**: Check available models by visiting `http://IP:11434/api/tags` and ensure the `OLLAMA_MODEL` matches exactly (e.g., `gemma4:latest`).
- **Embedding Support**: If you switch back to `OllamaEmbedding` and see a 501 error, it means the model does not support the `/api/embed` endpoint. The current configuration uses local embeddings to bypass this.
