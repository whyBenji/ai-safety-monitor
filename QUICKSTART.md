# Quick Start Guide

## Step 1: Setup Environment

### Install Dependencies

```bash
# Create virtual environment (if not already done)
python -m venv aiSafetyEnv
source aiSafetyEnv/bin/activate  # On Windows: aiSafetyEnv\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Setup API Keys

**For Hugging Face (required for Gemma):**
```bash
huggingface-cli login
# Enter your Hugging Face token when prompted
```

**For OpenAI (required for answer generation and OpenAI moderation):**
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or add to your .env file
```

## Step 2: Initialize Database

```bash
# For SQLite (easiest, no setup needed)
python -m monitor.storage.setup_db --database-url sqlite:///./ai_monitor.db

# For PostgreSQL (if using docker-compose)
python -m monitor.storage.setup_db --database-url postgresql+psycopg://myuser:mypassword@127.0.0.1:5433/mydb
```

## Step 3: Run the Pipeline

### Option A: Full Pipeline (Recommended)

Runs complete pipeline: Input Classification → Answer Generation → Output Classification

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier gemma \
  --answer-generator openai \
  --output-classifier gemma \
  --limit 10 \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/pipeline_results.json \
  --preview 5 \
  --verbose
```

### Option B: OpenAI-Only Pipeline

If you don't have Gemma set up, use OpenAI for everything:

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --openai-moderation-model omni-moderation-latest \
  --openai-answer-model gpt-4o-mini \
  --limit 10 \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/pipeline_results.json \
  --preview 5
```

### Option C: Input Classification Only

Skip answer generation and output classification:

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier gemma \
  --answer-generator none \
  --output-classifier none \
  --limit 10 \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/input_only.json \
  --preview 5
```

### Option D: Legacy Moderation (Backward Compatible)

The old input-only moderation still works:

```bash
python monitor/moderator/moderation_output.py \
  --provider gemma \
  --limit 10 \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/moderation_results.json \
  --preview 5
```

## Step 4: View Results in Dashboard

### Start the Dashboard

```bash
# Set database URL (if using SQLite)
export AI_SAFETY_MONITOR_DB=sqlite:///./ai_monitor.db

# Or for PostgreSQL
export AI_SAFETY_MONITOR_DB=postgresql+psycopg://myuser:mypassword@127.0.0.1:5433/mydb

# Start the dashboard
uvicorn monitor.dashboard.app:app --reload --port 8000
```

### Access Dashboard

Open your browser to: `http://127.0.0.1:8000`

You'll see:
- **Overview page**: List of all runs with metrics
- **Run detail page**: Full pipeline results for each run
  - Input classification results
  - Generated answers (if any)
  - Output classification results (if any)
  - Human review interface

## Common Configurations

### Minimal Test Run (Fast)

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --limit 3 \
  --database-url sqlite:///./ai_monitor.db \
  --preview 3
```

### Production Run (Full Pipeline)

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier gemma \
  --answer-generator openai \
  --output-classifier gemma \
  --gemma-base-model google/gemma-2b-it \
  --openai-answer-model gpt-4o-mini \
  --limit 1000 \
  --dataset-id allenai/real-toxicity-prompts \
  --split train \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/production_run.json \
  --verbose
```

### Using Custom Gemma Adapter

```bash
python monitor/pipeline/pipeline_cli.py \
  --input-classifier gemma \
  --gemma-base-model google/gemma-2b-it \
  --gemma-adapter-path /path/to/your/adapter \
  --answer-generator openai \
  --output-classifier gemma \
  --gemma-adapter-path /path/to/your/adapter \
  --limit 50 \
  --database-url sqlite:///./ai_monitor.db
```

## Troubleshooting

### "No module named 'monitor'"
Make sure you're running from the project root directory.

### "Hugging Face authentication required"
Run `huggingface-cli login` and provide your token.

### "OpenAI API key not found"
Set `export OPENAI_API_KEY="your-key"` or add to `.env` file.

### "Database schema error"
Run the database setup command again:
```bash
python -m monitor.storage.setup_db --database-url sqlite:///./ai_monitor.db
```

### Dashboard shows no data
Make sure the `AI_SAFETY_MONITOR_DB` environment variable matches the `--database-url` used when running the pipeline.

## Pipeline Flow

The complete pipeline works as follows:

1. **Input Classification**: Each prompt is classified as SAFE or TOXIC
2. **Answer Generation** (if input is SAFE): LLM generates an answer
3. **Output Classification** (if answer was generated): The generated answer is classified
4. **Storage**: All results are saved to database and JSON file
5. **Dashboard**: Human reviewers can inspect and override decisions

## Next Steps

- Review results in the dashboard
- Use human review to override machine decisions
- Export data from the database for analysis
- Fine-tune Gemma adapters for better classification
- Compare different model combinations

