from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import DataLoader, DatasetSplit
from src.eval_utils import evaluate_with_confusion
from src.model import MLPClassifier
from src.train_utils import load_checkpoint
from src.viz import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    with (out_dir / "classes.json").open("r", encoding="utf-8") as f:
        class_names = json.load(f)
    with (out_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    stats = np.load(out_dir / "norm_stats.npz")
    mean, std = stats["mean"], stats["std"]
    test_paths = [x[0] for x in splits["test"]]
    test_labels = np.array([int(x[1]) for x in splits["test"]], dtype=np.int64)
    test_split = DatasetSplit(test_paths, test_labels)

    model = MLPClassifier(
        input_dim=64 * 64 * 3,
        hidden_dim1=int(np.load(out_dir / "best_model.npz", allow_pickle=True)["meta_hidden1"]),
        hidden_dim2=int(np.load(out_dir / "best_model.npz", allow_pickle=True)["meta_hidden2"]),
        num_classes=len(class_names),
        activation=str(np.load(out_dir / "best_model.npz", allow_pickle=True)["meta_activation"]),
    )
    load_checkpoint(model, out_dir / "best_model.npz")

    test_loader = DataLoader(test_split, args.batch_size, mean, std, shuffle=False, image_size=64)
    acc, cm, errors = evaluate_with_confusion(model, test_loader, len(class_names))
    print(f"test_acc={acc:.4f}")
    print("confusion_matrix:")
    print(cm)
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"test_acc": acc, "num_errors": len(errors)}, f, indent=2)
    np.save(out_dir / "confusion_matrix.npy", cm)
    with (out_dir / "errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()

