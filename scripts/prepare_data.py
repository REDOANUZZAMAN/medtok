"""Thin wrapper: python scripts/prepare_data.py ... == python -m medical_tokenizer.data.prepare ..."""
from medical_tokenizer.data.prepare import main

if __name__ == "__main__":
    main()
