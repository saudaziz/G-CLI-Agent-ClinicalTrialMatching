# Agentic Clinical Trial Matching System

An AI-powered MVP that bridges legacy healthcare data and modern research protocols using **LangGraph**, **LlamaIndex**, and the **Model Context Protocol (MCP)**.

## Scenario: The Clinical Recruitment Bottleneck

A pharmaceutical research organization manages hundreds of active clinical trials. Currently, specialized medical coordinators manually review thousands of patient electronic health records (EHR) to determine if they meet strict "Inclusion/Exclusion" criteria buried in 200-page Clinical Trial Protocols (PDFs). 

The patient data resides in a legacy SQL-based Hospital Information System (HIS) that lacks modern APIs, creating a critical bottleneck in **Healthcare, Pharmaceuticals, and Biotechnology**. Delayed recruitment is the #1 reason trials fail, costing millions per day in lost patent life.

**Similar Scenarios:** Veteran affairs benefit adjudication, complex insurance underwriting, and legal discovery.

## The Agentic Design

### Agent Roles
- **The Researcher (Medical Analyst):** Uses RAG to parse the Trial Protocol PDF and an MCP Server to securely query the legacy patient database for specific lab results and ICD-10 codes.
- **The Orchestrator (Clinical Lead):** Receives the trial ID, sets recruitment goals, and manages the logic flow between data extraction and eligibility.
- **The Executor (Regulatory Reporter):** Compiles a "Match Justification Report" citing specific protocol pages and patient record timestamps.

### The Math of Efficiency
Given $P = 10,000$ patients in a database:
- **Human Manual Review:** $H_{time} = 30 \text{ mins/patient}$ → **5,000 hours**.
- **Agentic Review:** $A_{time} = 20 \text{ seconds/patient}$ → **~55 hours**.
- **Efficiency Ratio ($C_e$):** Roughly **150:1** compared to professional human rates vs. token costs.

## Technical Stack
- **Languages:** TypeScript (MCP Server), Python (Agent Logic).
- **Orchestration:** **LangGraph** (Stateful multi-agent workflows).
- **RAG Framework:** **LlamaIndex** (Summary & Vector indexing).
- **Connectivity:** **MCP (Model Context Protocol)** as a secure bridge to legacy SQL/HIS.
- **LLM Support:** Claude 3.5 Sonnet, Gemini 1.5 Flash, or local **Ollama (Gemma4)**.

## Setup & Configuration

### 1. MCP Server (Legacy Data Simulation)
```bash
cd mcp-server
npm install && npm run build
npm start
```

### 2. Agent System (Python Logic)
```bash
cd agent-system
pip install -r requirements.txt
```

### 3. Environment Configuration (`agent-system/.env`)
Configure your LLM provider and network settings:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://192.168.68.190:11434
OLLAMA_MODEL=gemma4:latest
```

## Running the MVP
```powershell
Set-Location agent-system; python main.py
```

## KPI & ROI
- **Enrollment Speed:** Targeted 30-40% reduction in recruitment phase.
- **Accuracy:** High-precision matching citing specific protocol pages.
- **ROI:** Potential savings of **$1M+ per trial** in operational overhead and accelerated time-to-market.

---
**License:** MIT  
**Author:** Saud Aziz
