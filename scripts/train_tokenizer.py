"""Thin wrapper: python scripts/train_tokenizer.py ... == python -m medical_tokenizer.train ..."""
from medical_tokenizer.train import main

if __name__ == "__main__":
    main()
