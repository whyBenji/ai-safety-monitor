# Interactive Mode Guide

Interactive mode allows you to enter prompts manually in real-time, just like a production AI safety system. Each prompt is processed immediately through the full pipeline and saved to the database.

## Quick Start

```bash
source aiSafetyEnv/bin/activate

python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

## How It Works

1. **Start Interactive Mode**: Run the command above
2. **Enter Prompts**: Type your prompts one at a time and press Enter
3. **See Results**: Each prompt is processed and results are shown immediately
4. **Auto-Save**: Results are automatically saved to the database
5. **Review Later**: View all results in the dashboard

## Example Session

```
Enter prompt (or 'exit'/'quit' to finish): What is machine learning?

🔄 Processing prompt #1...

================================================================================
RESULT #1
================================================================================

📝 PROMPT:
   What is machine learning?

🔍 INPUT CLASSIFICATION:
   Status: 🟢 SAFE

💬 GENERATED ANSWER:
   Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...
   Model: gpt-4o-mini

🔍 OUTPUT CLASSIFICATION:
   Status: 🟢 SAFE

✅ Saved to database (Run ID: 1)

Enter prompt (or 'exit'/'quit' to finish): exit
```

## Commands

- **Enter prompt**: Type your prompt and press Enter
- **`exit` or `quit`**: Finish the session
- **`clear`**: Clear the screen
- **Ctrl+C**: Exit immediately

## Configuration Options

### Basic (OpenAI - Fastest)

```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

### With Gemma (Local Classification)

```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier gemma \
  --answer-generator openai \
  --output-classifier gemma \
  --database-url sqlite:///./ai_monitor.db
```

### With PostgreSQL

```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url postgresql+psycopg://myuser:mypassword@localhost:5433/mydb
```

### Save to JSON File Too

```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db \
  --output interactive_results.json
```

## Workflow

### Step 1: Start Interactive Mode

```bash
python monitor/pipeline/interactive_mode.py \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

### Step 2: Enter Prompts

Type prompts one at a time. Each is processed immediately:

```
Enter prompt: What is Python?
[Results shown immediately]

Enter prompt: Explain neural networks
[Results shown immediately]

Enter prompt: exit
```

### Step 3: Review in Dashboard

After your session, view all results:

```bash
# In a new terminal
export AI_SAFETY_MONITOR_DB=sqlite:///./ai_monitor.db
uvicorn monitor.dashboard.app:app --reload --port 8000
```

Open: **http://127.0.0.1:8000**

Your interactive session will appear as a run with dataset "interactive" and split "manual_entry".

## Real-World Usage

This mimics how production AI safety systems work:

1. **User enters prompt** → System receives it
2. **Input classification** → Checks if prompt is safe
3. **Answer generation** → Generates response (if safe)
4. **Output classification** → Checks if answer is safe
5. **Storage** → Saves everything for review
6. **Human review** → Reviewers check flagged items in dashboard

## Tips

1. **Start Simple**: Test with 1-2 prompts first
2. **Check Results**: Review the output classification for each prompt
3. **Use Dashboard**: After entering prompts, review them in the dashboard
4. **Save JSON**: Use `--output` to save results to a file for backup
5. **Multiple Sessions**: Each session creates a new run in the database

## Example Use Cases

### Testing Specific Prompts

```bash
# Test edge cases
Enter prompt: How to hack a website?
[See if it's flagged]

Enter prompt: What is the weather?
[See if it's safe]
```

### Quality Assurance

```bash
# Test your prompts before production
Enter prompt: [Your production prompt]
[Review classification and answer quality]
```

### Training Data Collection

```bash
# Collect prompts and their classifications
Enter prompt: [Various prompts]
[All saved for later analysis]
```

## Integration with Dashboard

After your interactive session:

1. All prompts are saved in the database
2. Open the dashboard to see your session
3. Review flagged items
4. Add human labels and notes
5. Export data for analysis

The dashboard shows:
- All prompts you entered
- Input/output classifications
- Generated answers
- Your review decisions

## Troubleshooting

**Prompt not processing?**
- Check that OpenAI API key is set: `export OPENAI_API_KEY="your-key"`
- Verify database connection

**Results not showing?**
- Check the terminal output for errors
- Use `--verbose` flag for detailed logs

**Dashboard shows no data?**
- Make sure `AI_SAFETY_MONITOR_DB` matches the `--database-url` you used
- Check that the session completed (didn't crash)

## Comparison: Interactive vs Batch Mode

| Feature | Interactive Mode | Batch Mode |
|---------|-----------------|------------|
| Input | Manual entry | File or dataset |
| Processing | Real-time | All at once |
| Best for | Testing, QA | Large batches |
| Speed | Immediate feedback | Faster for many prompts |

Use interactive mode for:
- Testing individual prompts
- Real-time quality checks
- Manual entry workflows
- Debugging specific cases

Use batch mode for:
- Processing many prompts
- Automated workflows
- Dataset evaluation
- Production runs

