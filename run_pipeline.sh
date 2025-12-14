#!/bin/bash
# Helper script to run the pipeline with proper environment setup

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "aiSafetyEnv" ]; then
    echo "Activating virtual environment..."
    source aiSafetyEnv/bin/activate
else
    echo "Warning: Virtual environment 'aiSafetyEnv' not found."
    echo "Please create it with: python -m venv aiSafetyEnv"
    exit 1
fi

# Check if database URL is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [database-url] [additional-args...]"
    echo ""
    echo "Examples:"
    echo "  $0 sqlite:///./ai_monitor.db"
    echo "  $0 postgresql+psycopg://myuser:mypassword@localhost:5433/mydb"
    echo "  $0 sqlite:///./ai_monitor.db --limit 5 --verbose"
    exit 1
fi

DB_URL="$1"
shift  # Remove first argument, pass rest to pipeline_cli.py

# Run the pipeline
echo "Running pipeline with database: $DB_URL"
python monitor/pipeline/pipeline_cli.py \
    --input-classifier gemma \
    --answer-generator openai \
    --output-classifier gemma \
    --database-url "$DB_URL" \
    "$@"

