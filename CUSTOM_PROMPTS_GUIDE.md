# Using Custom Prompts

You can now use your own custom prompts instead of the default dataset. There are three ways to provide custom prompts:

## Method 1: Single Prompt from Command Line

Process a single prompt directly:

```bash
python monitor/pipeline/pipeline_cli.py \
  --prompt "What is artificial intelligence?" \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

## Method 2: Multiple Prompts from Command Line

Process multiple prompts by using `--prompt` multiple times:

```bash
python monitor/pipeline/pipeline_cli.py \
  --prompt "What is machine learning?" \
  --prompt "Explain neural networks" \
  --prompt "How does deep learning work?" \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

## Method 3: Prompts from a Text File (Recommended)

Create a text file with one prompt per line:

**Example: `my_prompts.txt`**
```
What is the capital of France?
Explain how machine learning works.
Write a story about a robot learning to paint.
What are the best practices for cybersecurity?
Tell me about renewable energy sources.
```

Then run:

```bash
python monitor/pipeline/pipeline_cli.py \
  --prompts-file my_prompts.txt \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db \
  --preview 5
```

You can also limit the number of prompts processed from the file:

```bash
python monitor/pipeline/pipeline_cli.py \
  --prompts-file my_prompts.txt \
  --limit 3 \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db
```

## Priority Order

The pipeline uses prompts in this priority order:
1. `--prompt` (command line prompts) - highest priority
2. `--prompts-file` (file with prompts)
3. `--dataset-id` (default: allenai dataset) - lowest priority

## Examples

### Quick Test with One Custom Prompt

```bash
python monitor/pipeline/pipeline_cli.py \
  --prompt "Hello, how are you?" \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url sqlite:///./ai_monitor.db \
  --preview 1
```

### Process Custom Prompts File with Gemma

```bash
python monitor/pipeline/pipeline_cli.py \
  --prompts-file examples/custom_prompts.txt \
  --input-classifier gemma \
  --answer-generator openai \
  --output-classifier gemma \
  --database-url sqlite:///./ai_monitor.db \
  --output Results/custom_run.json
```

### Multiple Prompts with PostgreSQL

```bash
python monitor/pipeline/pipeline_cli.py \
  --prompt "First prompt here" \
  --prompt "Second prompt here" \
  --prompt "Third prompt here" \
  --input-classifier openai \
  --answer-generator openai \
  --output-classifier openai \
  --database-url postgresql+psycopg://myuser:mypassword@localhost:5433/mydb
```

## File Format

The prompts file should be a plain text file with:
- One prompt per line
- Empty lines are ignored
- UTF-8 encoding
- No special formatting needed

**Example file:**
```
What is Python?
How do I learn programming?
Explain quantum computing.
What are the benefits of cloud computing?
```

## Tips

1. **Start Small**: Test with 1-2 prompts first before processing large files
2. **Use Files for Many Prompts**: For more than 5 prompts, use a file instead of command line
3. **Check Results**: Always use `--preview` to see results in the console
4. **Save Output**: Use `--output` to save results to JSON for later analysis
5. **View in Dashboard**: After running, view results in the dashboard at http://127.0.0.1:8000

## Viewing Results

After processing custom prompts:

1. **In Console**: Results are previewed automatically
2. **In JSON File**: Check the `--output` file (default: `pipeline_results.json`)
3. **In Dashboard**: 
   ```bash
   export AI_SAFETY_MONITOR_DB=sqlite:///./ai_monitor.db
   uvicorn monitor.dashboard.app:app --reload --port 8000
   ```
   Then open http://127.0.0.1:8000

The dashboard will show your custom prompts in the run list, marked as "custom_cli" or "custom_file" in the dataset column.

