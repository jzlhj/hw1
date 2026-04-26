from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DatasetSplit:
    paths: list[str]
    labels: np.ndarray


def discover_dataset(root: str | Path) -> tuple[list[str], dict[str, int], list[str], np.ndarray]:
    root = Path(root)
    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    paths: list[str] = []
    labels: list[int] = []
    for cls in class_names:
        cls_dir = root / cls
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(class_to_idx[cls])
    return class_names, class_to_idx, paths, np.array(labels, dtype=np.int64)


def stratified_split(
    paths: list[str],
    labels: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test : n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val :].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return (
        DatasetSplit([paths[i] for i in train_idx], labels[train_idx]),
        DatasetSplit([paths[i] for i in val_idx], labels[val_idx]),
        DatasetSplit([paths[i] for i in test_idx], labels[test_idx]),
    )


def load_image(path: str, image_size: int = 64) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def compute_mean_std(paths: list[str], image_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros((3,), dtype=np.float64)
    total_sq = np.zeros((3,), dtype=np.float64)
    count = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.reshape(-1, 3)
        total += arr.sum(axis=0)
        total_sq += (arr**2).sum(axis=0)
        count += arr.shape[0]
    mean = total / count
    var = total_sq / count - mean**2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)


class DataLoader:
    def __init__(
        self,
        split: DatasetSplit,
        batch_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        shuffle: bool = True,
        image_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)
        self.mean = mean.reshape(1, 1, 1, 3)
        self.std = std.reshape(1, 1, 1, 3)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(len(self.split.paths))
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            xs: list[np.ndarray] = []
            ys: list[int] = []
            for i in batch_idx:
                p = self.split.paths[int(i)]
                img = Image.open(p).convert("RGB")
                if img.size != (self.image_size, self.image_size):
                    img = img.resize((self.image_size, self.image_size))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = (arr - self.mean) / self.std
                xs.append(arr.reshape(-1))
                ys.append(int(self.split.labels[int(i)]))
            yield np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.int64)

    def __len__(self) -> int:
        return (len(self.split.paths) + self.batch_size - 1) // self.batch_size

