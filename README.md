# Multi-Subtask Customer Support AI Pipeline

An end-to-end, multi-turn customer support AI system built with **FastAPI**, **Streamlit**, **ChromaDB**, **Google Gemini**, and **Gemma 2B (Base & LoRA Fine-Tuned SLM)**.

The pipeline processes incoming customer queries across three coordinated subtasks:
1. **Subtask 1 — Sentiment Analysis**: Classifies query sentiment (`Positive`, `Neutral`, or `Negative`) via Gemini structured outputs.
2. **Subtask 2 — Escalation Detection**: Determines whether human intervention is required (`true`/`false`) via Gemini.
3. **Subtask 3 — Context-Aware Response Generation**: Generates policy-grounded answers across multi-turn conversations using a **Dynamic Router** (routing `Simple` queries to local Fine-Tuned Gemma 2B and `Complex` queries to Gemini), direct model selection, and dual-domain **RAG** (`ShopEase E-Commerce` vs. `AdTech Workplace`).

---

## Key Features

- **Gemini-Driven Subtasks 1 & 2**: Decouples sentiment and escalation classification to run reliably on Gemini with `tenacity` exponential-backoff retry logic (`3` max attempts) and REST fallback resilience.
- **Dynamic Query Router**: Automatically classifies query complexity (`Simple` vs. `Complex`) to balance local SLM speed with cloud LLM reasoning.
- **Dual-Domain RAG Knowledge Base**: Powered by `ChromaDB` and `SentenceTransformers`, supporting both **Vanilla Mode** (ShopEase E-Commerce) and **Workplace Solution Mode** (AdTech Solutions).
- **Multi-Turn Conversational Chatbot**: Maintains prior conversation turns for context-aware follow-up questions, complete with a **Dev UI** toggle in Streamlit to inspect or hide backend metadata.
- **Automated Evaluation & Dashboard**: Benchmarks end-to-end accuracy, policy-filtered grounding rate, latency, throughput, and **LLM-as-a-Judge** correctness scores (`0–10` rubric).

---

## Project Structure

```text
WILP_API_Assignment_2/
├── app/
│   ├── main.py                 # FastAPI server & /api/v1/customer-query endpoint
│   ├── config.py               # Environment & model configuration settings
│   ├── schemas.py              # Pydantic request/response & chat history schemas
│   └── services/
│       ├── gemini.py           # Gemini REST client, retry logic, Subtasks 1–3 & Router
│       ├── slm.py              # Local Gemma 2B (Base & LoRA Fine-Tuned) inference
│       └── rag.py              # ChromaDB dual-domain vector retrieval
├── data/
│   ├── test_dataset.json       # 25-case benchmark dataset with policy grounding flags
│   └── eval_results.json       # Generated benchmark metrics & itemized results
├── scripts/
│   └── evaluate_pipeline.py    # Automated benchmark & LLM-as-a-Judge evaluation runner
├── streamlit_app.py            # Multi-turn conversational UI with Dev UI toggle
├── eval_dashboard.py           # Streamlit analytics dashboard for evaluation results
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables (API keys & model config)
```

---

## 1. Clone the Repository

Pull the latest codebase from GitHub and navigate into the project directory:

```bash
git clone https://github.com/rouzif/WILP_API_Assignment_2.git
cd WILP_API_Assignment_2
```

---

## 2. Create and Activate a Virtual Environment (`venv`)

Create an isolated Python virtual environment (Python 3.10+ recommended):

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt / PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

---

## 3. Install Dependencies

Upgrade `pip` and install all required packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Environment Configuration

Create a `.env` file in the root directory (`WILP_API_Assignment_2/.env`) and configure your Gemini API key and default model:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_MODEL=gemini-2.5-flash-lite
```

*(Note: Ensure local Gemma 2B base/fine-tuned model weights are placed under `./models/` if running offline, or ensure Hugging Face access is available for initial cache download.)*

---

## 5. Run the FastAPI Backend

Start the FastAPI server using `uvicorn`:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **API Endpoint**: `http://127.0.0.1:8000/api/v1/customer-query`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`

---

## 6. Run the Streamlit Chatbot UI

Open a **new terminal window**, activate your `venv`, and launch the Streamlit conversational interface:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

### Using the UI
- **Dev UI Toggle**: Turn **ON** in the sidebar to inspect Subtask 1 (Sentiment), Subtask 2 (Escalation), RAG status, Router complexity decisions, and target model details; turn **OFF** for a clean customer-facing chat view.
- **Application Mode**: Switch between **Vanilla Mode** (ShopEase) and **Workplace Solution** (AdTech).
- **Active Model**: Choose `Gemini`, `Gemma 2B (Base)`, `Gemma 2B (Fine-Tuned)`, or `Dynamic Router`.
- **Multi-Turn Follow-Ups**: Ask follow-up questions naturally in the chat input; prior conversation turns are preserved automatically.

---

## 7. Run the Automated Evaluation & Dashboard

Ensure the **FastAPI backend is running** (`uvicorn app.main:app --port 8000`), then execute the benchmark script in a separate terminal:

```bash
source venv/bin/activate
python scripts/evaluate_pipeline.py
```

This runs all 25 test cases in `data/test_dataset.json` through the `/api/v1/customer-query` endpoint, evaluates policy grounding exclusively on policy-dependent queries (`requires_policy: true`), scores semantic correctness (`0–10`) using Gemini as an LLM Judge, and saves the report to `data/eval_results.json`.

### View the Evaluation Dashboard
Launch the Streamlit evaluation dashboard to visualize metrics and inspect per-query breakdowns:

```bash
streamlit run eval_dashboard.py
```