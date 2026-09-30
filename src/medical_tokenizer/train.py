"""python -m medical_tokenizer.train --config configs/image_vq.yaml [--set train.epochs=2]"""
from __future__ import annotations

from .trainers import train
from .utils.config import arg_parser, load_config


def main(argv=None) -> None:
    ap = arg_parser("Train a modality tokenizer (VQ / continuous autoencoder, or fit event tokens).")
    args = ap.parse_args(argv)
    if not args.config:
        ap.error("--config is required")
    run = train(load_config(args.config, args.set))
    print(f"done: {run.dir}")


if __name__ == "__main__":
    main()
