"""Thin wrapper: python scripts/run_downstream.py ... == python -m medical_tokenizer.downstream ..."""
from medical_tokenizer.downstream.__main__ import main

if __name__ == "__main__":
    main()
