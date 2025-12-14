#!/usr/bin/env python3
"""
Quick test script to verify the pipeline setup works.
Run this to test your installation before running the full pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from monitor.pipeline import PipelineService
        from monitor.providers.input_classifier import GemmaInputClassifier, OpenAIInputClassifier
        from monitor.providers.answer_generator import OpenAIAnswerGenerator
        from monitor.providers.output_classifier import GemmaOutputClassifier, OpenAIOutputClassifier
        from monitor.storage import ModerationRepository
        from schema import PipelineResult, Prompt
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_openai_key():
    """Test if OpenAI API key is set."""
    print("\nTesting OpenAI API key...")
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        print(f"✓ OpenAI API key found (starts with: {key[:10]}...)")
        return True
    else:
        print("✗ OPENAI_API_KEY not set (required for answer generation)")
        print("  Set it with: export OPENAI_API_KEY='your-key'")
        return False

def test_database():
    """Test database connection."""
    print("\nTesting database setup...")
    try:
        db_url = "sqlite:///./test_ai_monitor.db"
        repo = ModerationRepository(db_url)
        repo.create_schema()
        print(f"✓ Database schema created at: {db_url}")
        # Clean up test database
        if Path("test_ai_monitor.db").exists():
            Path("test_ai_monitor.db").unlink()
        return True
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False

def test_simple_pipeline():
    """Test a minimal pipeline run (OpenAI only, no Gemma)."""
    print("\nTesting simple pipeline (OpenAI only)...")
    try:
        from monitor.pipeline import PipelineService
        from monitor.providers.input_classifier import OpenAIInputClassifier
        from monitor.providers.answer_generator import OpenAIAnswerGenerator
        from monitor.providers.output_classifier import OpenAIOutputClassifier
        from monitor.prompts import load_prompts
        from schema import Prompt
        
        # Check OpenAI key
        if not os.environ.get("OPENAI_API_KEY"):
            print("  Skipping (OPENAI_API_KEY not set)")
            return None
        
        # Create a simple test prompt
        test_prompt = Prompt(text="Hello, how are you?", metadata={})
        
        # Build pipeline
        input_classifier = OpenAIInputClassifier()
        answer_generator = OpenAIAnswerGenerator()
        output_classifier = OpenAIOutputClassifier()
        
        service = PipelineService(
            input_classifier=input_classifier,
            answer_generator=answer_generator,
            output_classifier=output_classifier,
        )
        
        # Run on single prompt
        results = service.process_prompts([test_prompt])
        
        if results and len(results) > 0:
            result = results[0]
            print(f"✓ Pipeline test successful!")
            print(f"  Input flagged: {result.input_classification.flagged}")
            print(f"  Answer generated: {result.answer is not None}")
            if result.answer:
                print(f"  Answer preview: {result.answer.text[:50]}...")
                print(f"  Output flagged: {result.output_classification.flagged if result.output_classification else 'N/A'}")
            return True
        else:
            print("✗ Pipeline returned no results")
            return False
            
    except Exception as e:
        print(f"✗ Pipeline test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Safety Monitor - Setup Test")
    print("=" * 60)
    
    results = {
        "imports": test_imports(),
        "openai_key": test_openai_key(),
        "database": test_database(),
    }
    
    # Only test pipeline if OpenAI key is available
    if results["openai_key"]:
        results["pipeline"] = test_simple_pipeline()
    else:
        results["pipeline"] = None
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(r for r in results.values() if r is not None)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. Initialize database: python -m monitor.storage.setup_db --database-url sqlite:///./ai_monitor.db")
        print("  2. Run pipeline: python monitor/pipeline/pipeline_cli.py --limit 10 --database-url sqlite:///./ai_monitor.db")
        print("  3. Start dashboard: uvicorn monitor.dashboard.app:app --reload --port 8000")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

