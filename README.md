# HR Resume Analyzer: Agentic Evaluation System

## Project Overview
This project is an advanced, multi-agent AI system designed to automate HR resume screening while strictly enforcing data privacy. It leverages a graph-based orchestration workflow to coordinate specialized agents, ensuring secure, observable, and bias-free candidate evaluations.

## Architecture & Agent-Graph Overview
This system is built using **LangGraph** to manage state, routing, and looping.
*   **State (`AgentState`):** A shared message state tracking the candidate's analysis progress.
*   **Agents (Nodes):**
    *   **Screener Agent:** Extracts raw resume data and strictly enforces data protection by calling a PII masking tool before handing off the data.
    *   **Matcher Agent:** Analyzes the sanitized skills against job requirements and outputs a recommendation.
*   **Edges & Routing:** Features a **looping graph structure** where the Screener calls tools and processes the output repeatedly until data is sanitized. A conditional edge halts the graph at a **Human-in-the-Loop (HITL)** node for HR Manager approval before finalizing any decisions.

## Key Features (Capstone Requirements)
1.  **Agentic Reasoning:** Agents use actionable logic to execute real function calls (`extract_resume_data`, `mask_pii`).
2.  **Multi-Agent Coordination:** Distinct Screener and Matcher roles with a clear hierarchical handoff mechanism.
3.  **Security & Guardrails:** Includes a prompt-injection detection node that intercepts and blocks malicious overrides ("force hire"). Output guardrails automatically mask PII (Emails/Phones).
4.  **Observability:** Fully integrated with LangSmith for structured tracing and logs.
5.  **Production Readiness:** 
    *   State is persisted using LangGraph's `SqliteSaver`, surviving system restarts.
    *   Containerized using Docker and Docker Compose for simulated cloud deployment.

## Prerequisites & Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally to see the executed logs and failure/security paths: 
   ```bash

SDAIA Academy Link: https://github.com/SDAIAAcademy
   
   python agent.py
