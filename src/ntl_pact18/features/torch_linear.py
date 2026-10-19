"""GPU-accelerated member: multinomial linear softmax over sparse n-gram features.

This is the one member that genuinely runs on CUDA / MPS. It reuses the same
four style views as :mod:`ntl_pact18.features.stylegram` but optimises the
classifier with Adam on the accelerator instead of liblinear on the CPU, which
also makes it a useful ensemble partner (different inductive bias: L2 logistic
loss vs. hinge loss).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from ..device import DeviceInfo, import_torch
from ..labels import LABELS
from ..utils import normalize_rows
from .stylegram import function_text, shape_text


class TorchLinearMember:
    name = "torch_linear"

    def __init__(self, device_info: DeviceInfo, epochs: int = 300, lr: float = 0.05,
                 weight_decay: float = 1e-4, seed: int = 42,
                 char_max_features: int = 24_000, word_max_features: int = 16_000,
                 shape_max_features: int = 7_000, function_max_features: int = 5_000,
                 predict_batch: int = 512):
        self.device_info = device_info
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.caps = {
            "char": int(char_max_features),
            "word": int(word_max_features),
            "shape": int(shape_max_features),
            "function": int(function_max_features),
        }
        self.predict_batch = int(predict_batch)
        self.vectorizers = {}
        self.W = None
        self.b = None
        self.torch = None
        self.device = None
        self.train_history = []
        self._W_np = None
        self._b_np = None

    # ------------------------------------------------------------ pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        state["torch"] = None
        state["_device_str"] = str(self.device) if self.device is not None else None
        state["_W_np"] = None if self.W is None else self.W.detach().cpu().numpy()
        state["_b_np"] = None if self.b is None else self.b.detach().cpu().numpy()
        state["W"] = None
        state["b"] = None
        state["device"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.torch = import_torch()
        if self.torch is not None and self._W_np is not None:
            device = self.device_info.torch_device() if self.device_info else None
            device = device or self.torch.device("cpu")
            self.W = self.torch.from_numpy(self._W_np).to(device)
            self.b = self.torch.from_numpy(self._b_np).to(device)
            self.device = device

    # ------------------------------------------------------------- features
    def _view(self, key: str, text: str) -> str:
        if key == "shape":
            return shape_text(text)
        if key == "function":
            return function_text(text)
        return text

    def _build(self, texts: Sequence[str]) -> None:
        specs = {
            "char": dict(analyzer="char", ngram_range=(3, 5)),
            "word": dict(analyzer="word", ngram_range=(1, 2)),
            "shape": dict(analyzer="char", ngram_range=(2, 5)),
            "function": dict(analyzer="word", ngram_range=(1, 2)),
        }
        for key, spec in specs.items():
            vec = TfidfVectorizer(
                sublinear_tf=True, min_df=2, max_features=self.caps[key],
                dtype=np.float32, lowercase=False, **spec,
            )
            if spec["analyzer"] == "word":
                vec.set_params(token_pattern=r"(?u)\b\w+\b|[^\w\s]")
            vec.fit([self._view(key, t) for t in texts])
            self.vectorizers[key] = vec

    def _transform(self, texts: Sequence[str]) -> csr_matrix:
        blocks = [self.vectorizers[k].transform([self._view(k, t) for t in texts]) for k in self.vectorizers]
        return hstack(blocks, format="csr").astype(np.float32)

    def _dense(self, texts: Sequence[str]) -> np.ndarray:
        """Dense float32 design matrix.

        CUDA sparse kernels (both COO and CSR) raise
        ``cudaErrorIllegalAddress`` after ~100 iterations on this workload,
        so the GPU path uses cuBLAS on a dense matrix instead. With ~52k
        features the matrix is ~0.4 GB for the training split, which fits
        comfortably on an 8 GB laptop GPU.
        """
        X = self._transform(texts)
        return np.ascontiguousarray(X.toarray(), dtype=np.float32)

    # ------------------------------------------------------------------- API
    def _train_on(self, X_dense: np.ndarray, y_idx, device):
        """Manual Adam on the linear softmax (dense, cuBLAS/MPS friendly)."""
        torch = self.torch
        X = torch.from_numpy(X_dense).to(device)
        Xt = X.t().contiguous()
        y = y_idx.to(device)
        n_features = X.shape[1]
        n = X.shape[0]
        Y = torch.nn.functional.one_hot(y, num_classes=len(LABELS)).float()

        W = torch.zeros(n_features, len(LABELS), device=device)
        b = torch.zeros(len(LABELS), device=device)
        mW = torch.zeros_like(W)
        vW = torch.zeros_like(W)
        mb = torch.zeros_like(b)
        vb = torch.zeros_like(b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        history = []

        with torch.no_grad():
            for step in range(1, self.epochs + 1):
                logits = X @ W + b
                probs = torch.softmax(logits, dim=1)
                G = (probs - Y) / n
                gW = Xt @ G + self.weight_decay * W
                gb = G.sum(dim=0)

                mW.mul_(beta1).add_(gW, alpha=1 - beta1)
                vW.mul_(beta2).addcmul_(gW, gW, value=1 - beta2)
                mb.mul_(beta1).add_(gb, alpha=1 - beta1)
                vb.mul_(beta2).addcmul_(gb, gb, value=1 - beta2)

                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                W.addcdiv_(mW, vW.sqrt().div_(bc2 ** 0.5).add_(eps), value=-self.lr / bc1)
                b.addcdiv_(mb, vb.sqrt().div_(bc2 ** 0.5).add_(eps), value=-self.lr / bc1)

                if step % 50 == 0 or step == self.epochs:
                    loss = torch.nn.functional.cross_entropy(logits, y)
                    history.append(float(loss.detach().cpu()))
        return W, b, history

    def fit(self, texts: Sequence[str], y: Sequence[str],
            groups: Optional[Sequence[int]] = None) -> "TorchLinearMember":
        torch = import_torch()
        if torch is None:
            raise RuntimeError("torch is not importable; disable features.torch_linear")
        self.torch = torch
        torch.manual_seed(self.seed)
        device = self.device_info.torch_device() or torch.device("cpu")

        self._build(texts)
        X_dense = self._dense(texts)
        y_idx = torch.tensor([LABELS.index(v) for v in y], dtype=torch.long)

        self.device = device
        try:
            W, b, history = self._train_on(X_dense, y_idx, device)
        except RuntimeError as exc:
            if device.type != "cuda":
                raise
            print(f"[torch_linear] CUDA training failed ({str(exc)[:120]}); retrying on CPU")
            self.device = torch.device("cpu")
            W, b, history = self._train_on(X_dense, y_idx, self.device)

        self.W, self.b = W.detach(), b.detach()
        self.train_history = history
        return self

    def predict_proba(self, texts: Sequence[str],
                      groups: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("TorchLinearMember.predict_proba called before fit")
        torch = self.torch
        device = self.device or torch.device("cpu")
        W = self.W.to(device)
        b = self.b.to(device)
        texts = list(texts)
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.predict_batch):
                chunk = texts[start:start + self.predict_batch]
                X = torch.from_numpy(self._dense(chunk)).to(device)
                logits = X @ W + b
                out.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        return normalize_rows(np.vstack(out))
