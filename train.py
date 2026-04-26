from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import DataLoader, compute_mean_std, discover_dataset, stratified_split
from src.model import MLPClassifier
from src.train_utils import SGD, TrainConfig, evaluate, save_checkpoint, train_one_epoch
from src.viz import plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="EuroSAT_RGB")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr_decay", type=float, default=0.95)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names, _, paths, labels = discover_dataset(args.data_root)
    train_split, val_split, test_split = stratified_split(paths, labels, seed=args.seed)
    mean, std = compute_mean_std(train_split.paths, image_size=args.image_size)
    np.savez(out_dir / "norm_stats.npz", mean=mean, std=std)

    train_loader = DataLoader(
        train_split, args.batch_size, mean, std, shuffle=True, image_size=args.image_size, seed=args.seed
    )
    val_loader = DataLoader(
        val_split, args.batch_size, mean, std, shuffle=False, image_size=args.image_size, seed=args.seed
    )

    model = MLPClassifier(
        input_dim=args.image_size * args.image_size * 3,
        hidden_dim1=args.hidden1,
        hidden_dim2=args.hidden2,
        num_classes=len(class_names),
        activation=args.activation,
        seed=args.seed,
    )
    opt = SGD(model.parameters(), lr=args.lr)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        hidden_dim1=args.hidden1,
        hidden_dim2=args.hidden2,
        activation=args.activation,
        seed=args.seed,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, args.weight_decay)
        va_loss, va_acc = evaluate(model, val_loader, args.weight_decay)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(opt.lr)
        print(
            f"epoch={epoch:03d} lr={opt.lr:.6f} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_acc > best_val:
            best_val = va_acc
            save_checkpoint(
                model,
                out_dir / "best_model.npz",
                meta={
                    "best_val_acc": best_val,
                    "hidden1": args.hidden1,
                    "hidden2": args.hidden2,
                    "activation": args.activation,
                    "image_size": args.image_size,
                },
            )
        opt.lr = max(args.min_lr, opt.lr * args.lr_decay)

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with (out_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    split_data = {
        "train": list(zip(train_split.paths, train_split.labels.tolist())),
        "val": list(zip(val_split.paths, val_split.labels.tolist())),
        "test": list(zip(test_split.paths, test_split.labels.tolist())),
    }
    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False)

    plot_training_curves(history, out_dir / "training_curves.png")
    print(f"best_val_acc={best_val:.4f}")


if __name__ == "__main__":
    main()

