from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.viz import visualize_errors, visualize_first_layer_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--num_weights", type=int, default=16)
    p.add_argument("--num_errors", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ckpt = np.load(out_dir / "best_model.npz", allow_pickle=True)
    w1 = ckpt["W1"]
    visualize_first_layer_weights(w1, out_dir / "first_layer_weights.png", num_show=args.num_weights)

    with (out_dir / "classes.json").open("r", encoding="utf-8") as f:
        class_names = json.load(f)
    if (out_dir / "errors.json").exists():
        with (out_dir / "errors.json").open("r", encoding="utf-8") as f:
            errors = json.load(f)
        visualize_errors(errors, class_names, out_dir / "error_examples.png", max_items=args.num_errors)


if __name__ == "__main__":
    main()

