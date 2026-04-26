from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .autograd import Tensor, softmax_cross_entropy
from .data import DataLoader
from .model import MLPClassifier


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 128
    lr: float = 0.05
    lr_decay: float = 0.95
    min_lr: float = 1e-4
    weight_decay: float = 1e-4
    hidden_dim1: int = 256
    hidden_dim2: int = 128
    activation: str = "relu"
    seed: int = 42


class SGD:
    def __init__(self, params: list[Tensor], lr: float) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad


def l2_regularization(model: MLPClassifier) -> Tensor:
    out = Tensor(0.0, requires_grad=True)
    for p in model.parameters():
        out = out + (p * p).sum()
    return out


def accuracy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    pred = logits.argmax(axis=1)
    return float((pred == y).mean())


def train_one_epoch(
    model: MLPClassifier,
    loader: DataLoader,
    optimizer: SGD,
    weight_decay: float,
) -> tuple[float, float]:
    losses: list[float] = []
    accs: list[float] = []
    for x_np, y_np in loader:
        x = Tensor(x_np, requires_grad=False)
        logits = model.forward(x)
        ce = softmax_cross_entropy(logits, y_np)
        reg = l2_regularization(model) * weight_decay
        loss = ce + reg
        model.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.data))
        accs.append(accuracy_from_logits(logits.data, y_np))
    return float(np.mean(losses)), float(np.mean(accs))


def evaluate(model: MLPClassifier, loader: DataLoader, weight_decay: float = 0.0) -> tuple[float, float]:
    losses: list[float] = []
    accs: list[float] = []
    for x_np, y_np in loader:
        logits = model.predict_logits(x_np)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / probs.sum(axis=1, keepdims=True)
        ce = -np.log(np.clip(probs[np.arange(len(y_np)), y_np], 1e-12, 1.0)).mean()
        if weight_decay > 0:
            reg = 0.0
            for p in model.parameters():
                reg += float((p.data**2).sum())
            ce += weight_decay * reg
        losses.append(float(ce))
        accs.append(accuracy_from_logits(logits, y_np))
    return float(np.mean(losses)), float(np.mean(accs))


def save_checkpoint(
    model: MLPClassifier,
    path: str | Path,
    meta: dict[str, str | int | float] | None = None,
) -> None:
    state = model.state_dict()
    if meta:
        for k, v in meta.items():
            state[f"meta_{k}"] = np.array(v)
    np.savez(path, **state)


def load_checkpoint(model: MLPClassifier, path: str | Path) -> dict[str, str]:
    data = np.load(path, allow_pickle=True)
    state = {k: data[k] for k in data.files if not k.startswith("meta_")}
    model.load_state_dict(state)
    meta = {k[5:]: str(data[k]) for k in data.files if k.startswith("meta_")}
    return meta

