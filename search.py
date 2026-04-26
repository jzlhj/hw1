from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from src.data import DataLoader, compute_mean_std, discover_dataset, stratified_split
from src.model import MLPClassifier
from src.train_utils import SGD, evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="EuroSAT_RGB")
    p.add_argument("--out_dir", type=str, default="outputs_search")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, choices=["grid", "random"], default="grid")
    p.add_argument("--trials", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names, _, paths, labels = discover_dataset(args.data_root)
    train_split, val_split, _ = stratified_split(paths, labels, seed=args.seed)
    mean, std = compute_mean_std(train_split.paths)

    train_loader = DataLoader(train_split, args.batch_size, mean, std, shuffle=True, seed=args.seed)
    val_loader = DataLoader(val_split, args.batch_size, mean, std, shuffle=False, seed=args.seed)

    search_space = {
        "lr": [0.1, 0.05, 0.02],
        "hidden1": [128, 256],
        "hidden2": [64, 128],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "activation": ["relu", "tanh"],
    }

    if args.mode == "grid":
        configs = [
            dict(zip(search_space.keys(), vals))
            for vals in product(
                search_space["lr"],
                search_space["hidden1"],
                search_space["hidden2"],
                search_space["weight_decay"],
                search_space["activation"],
            )
        ]
    else:
        rng = np.random.default_rng(args.seed)
        configs = []
        for _ in range(args.trials):
            configs.append({k: rng.choice(v).item() if isinstance(rng.choice(v), np.generic) else rng.choice(v) for k, v in search_space.items()})

    results: list[dict[str, float | int | str]] = []
    best = {"val_acc": -1.0}
    for i, cfg in enumerate(configs, 1):
        model = MLPClassifier(
            input_dim=64 * 64 * 3,
            hidden_dim1=int(cfg["hidden1"]),
            hidden_dim2=int(cfg["hidden2"]),
            num_classes=len(class_names),
            activation=str(cfg["activation"]),
            seed=args.seed + i,
        )
        opt = SGD(model.parameters(), lr=float(cfg["lr"]))
        for _ in range(args.epochs):
            train_one_epoch(model, train_loader, opt, float(cfg["weight_decay"]))
            _, val_acc = evaluate(model, val_loader, float(cfg["weight_decay"]))
            opt.lr = max(1e-4, opt.lr * 0.95)
        result = {
            "trial": i,
            "lr": float(cfg["lr"]),
            "hidden1": int(cfg["hidden1"]),
            "hidden2": int(cfg["hidden2"]),
            "weight_decay": float(cfg["weight_decay"]),
            "activation": str(cfg["activation"]),
            "val_acc": float(val_acc),
        }
        results.append(result)
        print(result)
        if val_acc > best["val_acc"]:
            best = result
    results = sorted(results, key=lambda x: x["val_acc"], reverse=True)
    with (out_dir / "search_results.json").open("w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, ensure_ascii=False, indent=2)
    print("best:", best)


if __name__ == "__main__":
    main()

