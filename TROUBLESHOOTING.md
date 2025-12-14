# Troubleshooting Guide

## Common Errors and Solutions

### Error: `ModuleNotFoundError: No module named 'pydantic'` (or other modules)

**Solution:** Activate your virtual environment first:

```bash
source aiSafetyEnv/bin/activate
```

Then install dependencies if needed:
```bash
pip install -r requirements.txt
```

### Error: `ImportError: cannot import name 'PipelineService'`

**Solution:** Make sure you're running from the project root directory:
```bash
cd /Users/benji/Documents/Projects/ai-safety-monitor
python monitor/pipeline/pipeline_cli.py ...
```

### Error: Database connection failed

**For PostgreSQL:**
- Make sure PostgreSQL is running: `docker-compose up -d` (if using docker-compose)
- Check connection: `psql -h localhost -p 5433 -U myuser -d mydb`
- Verify password matches in the connection URL

**For SQLite:**
- Make sure the directory exists and is writable
- Check file permissions

### Error: `Hugging Face authentication required`

**Solution:**
```bash
huggingface-cli login
# Enter your Hugging Face token
```

### Error: `OpenAI API key not found`

**Solution:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Error: `Database schema error` or `table does not exist`

**Solution:** Initialize the database schema:
```bash
source aiSafetyEnv/bin/activate
python -m monitor.storage.setup_db --database-url YOUR_DATABASE_URL
```

### Error: Dashboard shows no data

**Solution:**
1. Make sure you've run the pipeline and saved data to the database
2. Verify the `AI_SAFETY_MONITOR_DB` environment variable matches the `--database-url` used in the pipeline:
   ```bash
   export AI_SAFETY_MONITOR_DB=sqlite:///./ai_monitor.db
   # Or for PostgreSQL:
   export AI_SAFETY_MONITOR_DB=postgresql+psycopg://myuser:mypassword@localhost:5433/mydb
   ```
3. Restart the dashboard

### Error: `command not found: python`

**Solution:** Use `python3` instead, or activate your virtual environment:
```bash
source aiSafetyEnv/bin/activate
python monitor/pipeline/pipeline_cli.py ...
```

### Error: Gemma model loading fails

**Possible causes:**
- Not authenticated with Hugging Face
- Insufficient memory (try smaller batch size)
- Network issues downloading model

**Solutions:**
- Run `huggingface-cli login`
- Use `--limit 1` for testing
- Try using OpenAI instead: `--input-classifier openai`

### Error: CUDA/GPU related errors

**Solution:** The code automatically falls back to CPU. If you see CUDA errors:
- Make sure PyTorch is installed correctly
- The code will work on CPU, just slower
- For GPU support, install CUDA-enabled PyTorch

## Quick Diagnostic Commands

### Test imports:
```bash
source aiSafetyEnv/bin/activate
python -c "from monitor.pipeline import PipelineService; print('OK')"
```

### Test database connection:
```bash
source aiSafetyEnv/bin/activate
python -c "from monitor.storage import ModerationRepository; r = ModerationRepository('sqlite:///./test.db'); r.create_schema(); print('OK')"
```

### Test OpenAI connection:
```bash
source aiSafetyEnv/bin/activate
export OPENAI_API_KEY="your-key"
python -c "from openai import OpenAI; c = OpenAI(); print('OK')"
```

## Getting Help

If you're still stuck:
1. Run with `--verbose` flag to see detailed logs
2. Check the error message carefully - it usually tells you what's missing
3. Make sure you're in the project root directory
4. Ensure virtual environment is activated
5. Verify all prerequisites are installed

