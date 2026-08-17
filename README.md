# AI-Powered Customer Support Agent

An AI-powered customer support copilot that drafts ticket replies using retrieval-augmented generation over a knowledge base, long-term customer memory (Mem0), and selective tool-calling — with input/output guardrails and request tracing. Served via FastAPI with a Streamlit dashboard.

## Features

- **Draft generation agent** (`copilot_service.py`) — a LangChain agent (`create_agent`) backed by Groq, with a LangGraph `InMemorySaver` checkpointer for per-conversation state.
- **Auto-updating model selection** (`integrations/llm/groq_models.py`, wired through `Settings.effective_groq_model()`) — fetches Groq's live model list at runtime and picks the best currently available free model, instead of relying on a hardcoded model ID that can silently break once Groq retires it. Every component that needs the LLM (the copilot agent, the guardrails scope classifier, Mem0's internal LLM) resolves the model through this single method, so there's one source of truth.
- **Retrieval-augmented knowledge base** — a ChromaDB-backed vector store (`chroma_kb.py`) over Markdown knowledge-base articles (banking FAQs, KYC rules, charges, ATM withdrawal policy, etc. in `knowledge_base/`).
- **Long-term customer memory, scoped two ways** — `mem0_store.py` uses Mem0 with a separate Chroma store to remember customer-specific context across tickets, scoped both per-customer (by email) and per-company, so a fact learned from one ticket can inform others from the same organization.
- **Selective, verified tool-calling** — the agent only calls tools (`lookup_customer_plan`, `lookup_open_ticket_load`) when the ticket text actually warrants it (plan/SLA questions, ticket-load questions); results are prefetched and injected into the prompt as "verified tool findings" rather than trusted blindly.
- **Input/output guardrails** (`guardrails_service.py`) — PII redaction (account numbers, cards, emails, phone numbers), scope classification (keeps the agent from answering off-topic requests), toxicity and forbidden-promise checks on generated drafts, with escalation to a human when a check fails. Uses [Guardrails AI](https://github.com/guardrails-ai/guardrails) hub validators when installed, and falls back to built-in regex-based validators when it isn't — so the service works either way.
- **Request tracing** (`observability/tracer.py`) — every LLM call is logged as a structured span (prompt, response, latency, knowledge hits, tool calls, guardrail outcomes) to daily JSONL files under `data/traces/`, with a `NoOpTracer` used when tracing is disabled.
- **Multi-layer fallback generation** — if the agent's tool-calling path returns empty content, the service falls back to a direct LLM call with the same context, and if that also fails, a deterministic templated response — so draft generation degrades gracefully instead of erroring out.
- **Evaluation harness** (`evals/`) — a golden dataset of test tickets (`evals/dataset/golden.json`) evaluated with Ragas/DeepEval-style metrics, producing JSON and Markdown reports under `evals/reports/`.
- **Ticket / customer / draft persistence** via SQLite repositories (`repositories/sqlite/`).
- **FastAPI backend** with routers for tickets, drafts, knowledge, memory, and health checks (`api/routers/`).
- **Streamlit dashboard** (`app.py`) for agents to view tickets and generated drafts.
- **Automated tests** (`tests/`, `evals/`) run via `pytest`, wired into CI.
- Docker, docker-compose, and GitHub Actions workflows for CI and EC2 deployment.

## Architecture

```
customer_support_agent/
├── api/                      # FastAPI app factory, dependencies, routers
├── core/settings.py          # pydantic-settings config (env-driven)
├── services/
│   ├── copilot_service.py       # the agent: RAG + memory + tools + guardrails + tracing
│   ├── draft_service.py
│   ├── guardrails_service.py    # PII redaction, scope/toxicity/promise checks
│   └── knowledge_service.py
├── repositories/sqlite/      # tickets, customers, drafts persistence
├── observability/tracer.py   # structured JSONL request tracing
├── integrations/
│   ├── rag/chroma_kb.py         # knowledge base retrieval
│   ├── memory/mem0_store.py     # customer + company-scoped memory
│   ├── llm/groq_models.py       # live Groq model discovery
│   └── tools/support_tools.py   # tools available to the agent
└── schemas/api.py            # request/response models

evals/                        # golden dataset + Ragas/DeepEval evaluation harness
```

## Requirements

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) (recommended — there's a `uv.lock`)
- A [Groq API key](https://console.groq.com/keys)
- A [Google AI Studio API key](https://aistudio.google.com/) for Gemini embeddings
- Optional: [Guardrails AI](https://github.com/guardrails-ai/guardrails) for ML-based PII/toxicity/scope validators (falls back to built-in regex validators if not installed)

## Installation

```bash
git clone https://github.com/Akditi/support-agent-memory-toolcalling.git
cd support-agent-memory-toolcalling
uv sync
```

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your-groq-key
# GROQ_MODEL=openai/gpt-oss-120b   # optional — see note below
GOOGLE_API_KEY=your-google-key
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001

# Optional toggles (all default to enabled/true)
GUARDRAILS_ENABLED=true
TRACE_ENABLED=true
```

> **Model selection:** `GROQ_MODEL` is optional. If it's left unset, `Settings.effective_groq_model()` queries Groq's `/openai/v1/models` endpoint at runtime, filters it down to chat-capable models, and picks the best one currently available (cached for an hour). This means the copilot, the guardrails scope classifier, and Mem0 all keep working automatically even after Groq deprecates a model — set `GROQ_MODEL` explicitly only if you want to pin a specific model instead.

## Usage

Run the API:

```bash
uv run python main.py
```

Run the dashboard (in a second terminal):

```bash
uv run streamlit run app.py
```

The API docs are available at `http://localhost:8000/docs` once the server is running.

## Testing & Evaluation

```bash
uv run pytest -q
```

The `evals/` suite runs the agent against a golden dataset and scores responses with Ragas/DeepEval metrics; results are written to `evals/reports/latest.json` and `evals/reports/latest.md`. Run the full evaluation with:

```bash
uv run pytest evals/ -m full_eval
```

## Deployment

- `Dockerfile` / `docker-compose.yml` for containerized runs.
- `.github/workflows/ci.yml` runs tests on every push/PR.
- `.github/workflows/deploy-ec2.yml` deploys to EC2 — see `docs/EC2_deployment_flow.md` for the full flow.

## Tech stack

FastAPI · LangChain · LangGraph · langchain-groq · Mem0 · ChromaDB · Groq (LLM inference) · Google Gemini (embeddings) · Guardrails AI · Ragas · DeepEval · Streamlit · SQLite

## Credits

This project is based on a project idea/tutorial from [Krish Naik's Projects](https://www.krishnaik.in/projects). I built and substantially extended it as a hands-on learning project — including the auto-updating Groq model selection, guardrails, request tracing, and evaluation harness described above.