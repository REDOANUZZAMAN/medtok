"""Thin wrapper: python scripts/run_pipeline.py ... == python -m medical_tokenizer.pipeline ..."""
from medical_tokenizer.pipeline import main

if __name__ == "__main__":
    main()
