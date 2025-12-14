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

### Repository Layout

```
monitor/
  pipeline/              # Pipeline orchestrator
    - pipeline_cli.py         # Batch processing CLI
    - interactive_mode.py     # Interactive real-time mode
    - pipeline_service.py      # Core pipeline logic
  providers/             # Classification and generation providers
    - input_classifier.py      # Input classification (Gemma/OpenAI)
    - answer_generator.py      # Answer generation (OpenAI)
    - output_classifier.py     # Output classification (Gemma/OpenAI)
  prompts/               # Prompt loading utilities
    - loading_prompts.py       # Dataset and custom prompt loaders
  storage/               # Database models and repository
    - models.py               # SQLAlchemy models
    - repository.py            # Database operations
    - setup_db.py             # Database initialization
  dashboard/             # Human-in-the-loop dashboard
    - app.py                  # FastAPI application
    - templates/              # HTML templates
  moderator/             # Legacy code (backward compatibility only)
toxic_gemma_classifier.py    # Gemma classifier implementation
schema.py                     # Pydantic data models
migrations/                   # Alembic database migrations
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

#### Option 1: Interactive Mode (Recommended for Testing)

Enter prompts manually in real-time, just like a production system:

```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

Then type prompts one at a time. Each prompt is processed immediately and saved to the database.

#### Option 2: Batch Mode with Custom Prompts

Process your own prompts from command line or file:

**Single prompt:**
```bash
python monitor/pipeline/pipeline_cli.py \
  --prompt "What is artificial intelligence?" \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

**Multiple prompts:**
```bash
python monitor/pipeline/pipeline_cli.py \
  --prompt "First question" \
  --prompt "Second question" \
  --prompt "Third question" \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

**From file (one prompt per line):**
```bash
python monitor/pipeline/pipeline_cli.py \
  --prompts-file my_prompts.txt \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

#### Option 3: Batch Mode with Dataset

Process prompts from a Hugging Face dataset:

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier gemma \
  --answer-generator openai \
  --output-classifier gemma \
  --limit 100 \
  --dataset-id allenai/real-toxicity-prompts \
  --database-url sqlite:///./ai_monitor.db \
  --preview 10
```

#### Key Configuration Options

| Flag | Description | Default |
| --- | --- | --- |
| `--input-classifier {gemma,openai}` | Input classification backend | `gemma` |
| `--answer-generator {openai,none}` | Answer generation backend | `openai` |
| `--output-classifier {gemma,openai,none}` | Output classification backend | `gemma` |
| `--prompt TEXT` | Custom prompt (can use multiple times) | None |
| `--prompts-file PATH` | File with custom prompts (one per line) | None |
| `--gemma-base-model` | Gemma model checkpoint | `google/gemma-2b-it` |
| `--openai-answer-model` | OpenAI model for answers | `gpt-4o-mini` |
| `--database-url` | Database connection string | None |
| `--output` | JSON output file path | `pipeline_results.json` |
| `--preview` | Number of results to preview | `5` |

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

### Complete Workflow Example

**Step 1: Process Prompts (Interactive Mode)**
```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```
Enter prompts one by one. Each is processed and saved immediately.

**Step 2: Review in Dashboard**
```bash
export AI_SAFETY_MONITOR_DB=sqlite:///./ai_monitor.db
uvicorn monitor.dashboard.app:app --reload --port 8000
```
Open http://127.0.0.1:8000 to review all prompts, flag issues, and add human labels.

**Step 3: Review Flagged Items**
- Click "Only flagged" to focus on problematic results
- Review each flagged prompt and generated answer
- Mark as SAFE or TOXIC with notes
- All reviews are saved instantly

---

### Quick Examples

**Interactive mode with Gemma:**
```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier gemma \
  --answer-generator openai \
  --output-classifier gemma \
  --database-url sqlite:///./ai_monitor.db
```

**Process custom prompts from file:**
```bash
python monitor/pipeline/pipeline_cli.py \
  --prompts-file my_prompts.txt \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

**OpenAI-only (fastest, no GPU needed):**
```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

### Customization

- **Gemma Adapters:** Use `--gemma-adapter-path` to load fine-tuned LoRA adapters
- **Custom Prompts:** Use `--prompt` or `--prompts-file` for your own prompts
- **Databases:** Use PostgreSQL for production: `--database-url postgresql+psycopg://...`
- **Models:** Configure different OpenAI models with `--openai-answer-model` and `--openai-moderation-model`

---

### Troubleshooting

**Common Issues:**

- **"No module named 'pydantic'" or similar**: Activate virtual environment: `source aiSafetyEnv/bin/activate`
- **"Column does not exist"**: Run migrations: `alembic -x database_url=... upgrade head`
- **"OpenAI API key not found"**: Set `export OPENAI_API_KEY="your-key"`
- **"Hugging Face authentication required"**: Run `huggingface-cli login`
- **Dashboard shows no data**: Ensure `AI_SAFETY_MONITOR_DB` matches the `--database-url` used in pipeline
- **Gemma model loading fails**: Check Hugging Face authentication and available memory
- **Port already in use**: Use different port: `uvicorn ... --port 8001`

**Database Issues:**

- **SQLite**: Works out of the box, no setup needed
- **PostgreSQL**: Ensure database exists and migrations are run: `alembic -x database_url=... upgrade head`
- **Schema errors**: Run `python -m monitor.storage.setup_db --database-url <URL>` for new databases

---

## Additional Documentation

- **MIGRATIONS.md** - Database migration guide with Alembic
- **INTERACTIVE_MODE_GUIDE.md** - Detailed guide for interactive mode
- **CUSTOM_PROMPTS_GUIDE.md** - Guide for using custom prompts

---

The combination of Gemma-based automation and human sign-off makes it easy to iterate on AI safety policies while keeping a verifiable audit trail. Contributions are welcome!
