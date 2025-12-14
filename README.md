## AI Safety Monitor

Complete AI safety pipeline with input classification, answer generation, and output classification, powered by Gemma and OpenAI, along with a human-in-the-loop dashboard for auditing model behavior.

The project implements a full safety pipeline:
1. **Input Classification** - Classify input prompts for safety/toxic content
2. **Answer Generation** - Generate answers using LLMs (only if input is safe)
3. **Output Classification** - Classify generated answers for safety/toxic content
4. **Human Review** - Dashboard for human reviewers to audit and override machine decisions

The pipeline ingests prompts (default: [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)), classifies them with Google's `gemma-2b-it` (with optional LoRA adapters) or OpenAI's moderation API, generates answers if inputs are safe, classifies the outputs, stores all results in a database, and surfaces decisions through an interactive FastAPI dashboard where reviewers can accept/reject every flag and leave notes.

---

### Highlights

- **Gemma-powered classifier** – `toxic_gemma_classifier.py` loads `google/gemma-2b-it` (or a fine-tuned adapter) and emits SAFE/TOXIC decisions with quantization when `bitsandbytes` is available.
- **Pluggable providers** – the moderation runner supports both the on-prem Gemma classifier and the OpenAI Moderations API for comparison studies.
- **Persistent telemetry** – runs, flags, raw payloads, and logs are saved via SQLAlchemy (SQLite or Postgres). Each record tracks human review status plus reviewer notes.
- **Human-in-the-loop dashboard** – `monitor/dashboard/app.py` delivers a FastAPI + Jinja UI to monitor runs, inspect individual prompts, and record SAFE/TOXIC overrides in real time.
- **CLI friendly** – a single command pulls prompts, classifies them, saves a JSON export, and updates the database for later review.

---

### Repository layout

```
monitor/
  pipeline/…          # Pipeline orchestrator (input -> answer -> output)
  moderator/…         # Legacy moderation service (backward compatibility)
  providers/…         # Pipeline components:
    - input_classifier.py   # Input classification (Gemma/OpenAI)
    - answer_generator.py   # Answer generation (OpenAI)
    - output_classifier.py  # Output classification (Gemma/OpenAI)
  prompts/…           # Dataset loader (Hugging Face Datasets)
  storage/…           # SQLAlchemy models & repository utilities
  dashboard/…         # FastAPI app + templates for HITL review
toxic_gemma_classifier.py  # Gemma + LoRA classification helper
requirements.txt           # Runtime dependencies
```

---

### Prerequisites

- Python 3.10+
- Access to the Gemma checkpoints on Hugging Face (`huggingface-cli login` before running the classifier)
- (Optional) OpenAI API key if you want to compare against `omni-moderation-latest`
- A database URL (SQLite by default, Postgres supported via SQLAlchemy)

The Gemma pipeline benefits from a GPU, but it also runs on CPU for small batches. When `bitsandbytes` is available the model automatically loads in 4-bit precision; otherwise it falls back to `torch.float16`.

---

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Authenticate with Hugging Face once so the Gemma weights can be pulled
huggingface-cli login
```

If you prefer Postgres, `docker-compose.yml` already exposes a ready-to-use database at `postgresql+psycopg://myuser:mypassword@127.0.0.1:5433/mydb`. Otherwise, SQLite works out of the box (e.g. `sqlite:///./ai_monitor.db`).

Initialize your database tables and run migrations:

```bash
# For new databases, create schema
python -m monitor.storage.setup_db --database-url sqlite:///./ai_monitor.db

# For existing databases, run Alembic migrations
alembic -x database_url=sqlite:///./ai_monitor.db upgrade head
# Or for PostgreSQL:
alembic -x database_url=postgresql+psycopg://myuser:mypassword@localhost:5433/mydb upgrade head
```

---

### Running the AI Safety Pipeline

The CLI runs the complete pipeline: input classification → answer generation → output classification.

**New Pipeline (Recommended):**

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier gemma \
  --answer-generator openai \
  --output-classifier gemma \
  --gemma-base-model google/gemma-2b-it \
  --openai-answer-model gpt-4o-mini \
  --limit 100 \
  --dataset-id allenai/real-toxicity-prompts \
  --split train \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/pipeline_results.json \
  --preview 10
```

Key switches:

| Flag | Description |
| --- | --- |
| `--input-classifier {gemma,openai}` | Input classification backend. Default is Gemma. |
| `--answer-generator {openai,none}` | Answer generation backend. Use 'none' to skip. |
| `--output-classifier {gemma,openai,none}` | Output classification backend. Use 'none' to skip. |
| `--gemma-base-model` | Hugging Face checkpoint for Gemma classifiers (works with LoRA adapters via `--gemma-adapter-path`). |
| `--openai-answer-model` | OpenAI model for answer generation (default: gpt-4o-mini). |
| `--openai-moderation-model` | OpenAI moderation model for input/output classification (default: omni-moderation-latest). |
| `--database-url` | SQLAlchemy URL for persisting runs/logs/results (SQLite or Postgres). |
| `--output` | JSON file containing the structured pipeline results. |

**Legacy Moderation (Input-only, backward compatibility):**

```bash
python monitor/moderator/moderation_output.py \
  --provider gemma \
  --gemma-base-model google/gemma-2b-it \
  --limit 100 \
  --dataset-id allenai/real-toxicity-prompts \
  --split train \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/moderation_results.json \
  --preview 10
```

The pipeline runner prints a short preview, writes the JSON dump to disk, and records the complete run in the database (including logs). You can re-run the command with different models/providers to compare machine behavior.

---

### Human-in-the-loop dashboard

The dashboard surfaces run metrics and lets reviewers approve/reject every flagged prompt.

1. Point the dashboard at the same database:

   ```bash
   export AI_SAFETY_MONITOR_DB=sqlite:///./ai_monitor.db
   ```

2. Launch FastAPI with Uvicorn:

   ```bash
   uvicorn monitor.dashboard.app:app --reload --port 8000
   ```

3. Open `http://127.0.0.1:8000/` to see recent runs, totals, and per-run detail pages.

From a run detail page you can:

- Inspect all prompts with their complete pipeline results (input classification, generated answers, output classification)
- Filter to see only flagged items (input or output)
- Expand the raw provider payloads for auditing
- Tag each result as **SAFE** or **TOXIC** (for input, output, or both)
- Add reviewer notes (stored in the `moderation_results` table)

Each review instantly updates the database, so multiple reviewers can collaborate simultaneously. The dashboard shows:
- **Input Classification**: Whether the input prompt was flagged as toxic
- **Generated Answer**: The answer generated by the LLM (if input was safe)
- **Output Classification**: Whether the generated answer was flagged as toxic
- **Human Review**: Override decisions with human labels and notes

---

### Suggested review flow

1. Run the classifier periodically (or whenever new prompts arrive) using the CLI command above.
2. Point the dashboard to the same DB file/server.
3. Triage flagged prompts via the dashboard:
   - Click “Only flagged” to focus on potential violations.
   - Record SAFE/TOXIC decisions, adding notes for escalations.
4. Export downstream analytics by querying the `moderation_results` table (machine labels, human overrides, timestamps, and notes are all stored there).

---

### Customization tips

- **Adapters:** point `--gemma-adapter-path` to any PEFT LoRA directory to swap in a fine-tuned classifier.
- **Datasets:** adapt `monitor/prompts/loading_prompts.py` to sample from your own corpus or augment metadata.
- **Databases:** update `AI_SAFETY_MONITOR_DB` and `--database-url` to reuse production-grade Postgres/CloudSQL instances.
- **Providers:** implement the `ClassifierProvider`/`ModerationProvider` protocols to plug in alternative moderation APIs.

---

### Troubleshooting

- The Gemma checkpoints require Hugging Face access grants; ensure `huggingface-cli whoami` succeeds before running the CLI.
- On Apple Silicon, `bitsandbytes` is optional. The classifier automatically falls back to half-precision tensors if quantization is unavailable.
- If you switch databases, rerun `python -m monitor.storage.setup_db --database-url <URL>` to ensure the new instance has the latest schema (including human review columns).
- Dashboard not showing data? Confirm the `AI_SAFETY_MONITOR_DB` value matches the `--database-url` used by the CLI run and that the server user has read/write permissions.

---

The combination of Gemma-based automation and human sign-off makes it easy to iterate on AI safety policies while keeping a verifiable audit trail. Contributions are welcome!
