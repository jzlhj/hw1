from __future__ import annotations

import numpy as np

from .autograd import Tensor


def _xavier(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        self.activation = activation.lower()
        self.rng = np.random.default_rng(seed)
        self.W1 = Tensor(_xavier(input_dim, hidden_dim1, self.rng), requires_grad=True, name="W1")
        self.b1 = Tensor(np.zeros((1, hidden_dim1), dtype=np.float32), requires_grad=True, name="b1")
        self.W2 = Tensor(_xavier(hidden_dim1, hidden_dim2, self.rng), requires_grad=True, name="W2")
        self.b2 = Tensor(np.zeros((1, hidden_dim2), dtype=np.float32), requires_grad=True, name="b2")
        self.W3 = Tensor(_xavier(hidden_dim2, num_classes, self.rng), requires_grad=True, name="W3")
        self.b3 = Tensor(np.zeros((1, num_classes), dtype=np.float32), requires_grad=True, name="b3")

    def _act(self, x: Tensor) -> Tensor:
        if self.activation == "relu":
            return x.relu()
        if self.activation == "sigmoid":
            return x.sigmoid()
        if self.activation == "tanh":
            return x.tanh()
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: Tensor) -> Tensor:
        h1 = self._act(x @ self.W1 + self.b1)
        h2 = self._act(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def parameters(self) -> list[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def predict_logits(self, x_np: np.ndarray) -> np.ndarray:
        x = Tensor(x_np, requires_grad=False)
        return self.forward(x).data

    def state_dict(self) -> dict[str, np.ndarray | str]:
        return {
            "W1": self.W1.data,
            "b1": self.b1.data,
            "W2": self.W2.data,
            "b2": self.b2.data,
            "W3": self.W3.data,
            "b3": self.b3.data,
            "activation": self.activation,
        }

    def load_state_dict(self, state: dict[str, np.ndarray | str]) -> None:
        self.W1.data = np.array(state["W1"], dtype=np.float32)
        self.b1.data = np.array(state["b1"], dtype=np.float32)
        self.W2.data = np.array(state["W2"], dtype=np.float32)
        self.b2.data = np.array(state["b2"], dtype=np.float32)
        self.W3.data = np.array(state["W3"], dtype=np.float32)
        self.b3.data = np.array(state["b3"], dtype=np.float32)
        self.activation = str(state.get("activation", self.activation))

