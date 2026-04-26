from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _to_array(x: np.ndarray | float | int) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return np.array(x, dtype=np.float32)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


@dataclass
class Tensor:
    data: np.ndarray
    requires_grad: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        self.data = _to_array(self.data)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set()

    def __hash__(self) -> int:
        return id(self)

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Grad must be provided for non-scalar tensors.")
            grad = np.ones_like(self.data, dtype=np.float32)
        elif not isinstance(grad, np.ndarray):
            grad = np.array(grad, dtype=np.float32)

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = grad.astype(np.float32, copy=False)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: Tensor | float | int) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __radd__(self, other: Tensor | float | int) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        return self + (-other)

    def __rsub__(self, other: Tensor | float | int) -> Tensor:
        return other + (-self)

    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rmul__(self, other: Tensor | float | int) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def pow(self, p: float) -> Tensor:
        out = Tensor(np.power(self.data, p), self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad * (p * np.power(self.data, p - 1.0))

        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad
                if axis is None:
                    grad = np.broadcast_to(grad, self.data.shape)
                else:
                    if not keepdims:
                        axes = (axis,) if isinstance(axis, int) else axis
                        for ax in sorted(axes):
                            grad = np.expand_dims(grad, axis=ax)
                    grad = np.broadcast_to(grad, self.data.shape)
                self.grad += grad

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(0.0, self.data), self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        out._prev = {self}
        return out

    def sigmoid(self) -> Tensor:
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        out._prev = {self}
        return out

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad * (1.0 - t**2)

        out._backward = _backward
        out._prev = {self}
        return out


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    targets = targets.astype(np.int64, copy=False)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    n = logits.data.shape[0]
    loss_value = -np.log(np.clip(probs[np.arange(n), targets], 1e-12, 1.0)).mean()
    out = Tensor(np.array(loss_value, dtype=np.float32), logits.requires_grad)

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1.0
        grad /= n
        logits.grad += grad * float(out.grad)

    out._backward = _backward
    out._prev = {logits}
    return out

