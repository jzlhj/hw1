from __future__ import annotations

import numpy as np

from .data import DataLoader
from .model import MLPClassifier


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def evaluate_with_confusion(
    model: MLPClassifier,
    loader: DataLoader,
    num_classes: int,
) -> tuple[float, np.ndarray, list[tuple[str, int, int]]]:
    ys: list[int] = []
    ps: list[int] = []
    errors: list[tuple[str, int, int]] = []
    all_paths = loader.split.paths
    offset = 0
    for x_np, y_np in loader:
        logits = model.predict_logits(x_np)
        pred = logits.argmax(axis=1)
        ys.extend(y_np.tolist())
        ps.extend(pred.tolist())
        batch_paths = all_paths[offset : offset + len(y_np)]
        for pth, gt, pd in zip(batch_paths, y_np.tolist(), pred.tolist()):
            if gt != pd:
                errors.append((pth, gt, pd))
        offset += len(y_np)
    y_true = np.array(ys, dtype=np.int64)
    y_pred = np.array(ps, dtype=np.int64)
    acc = float((y_true == y_pred).mean())
    return acc, confusion_matrix(y_true, y_pred, num_classes), errors

