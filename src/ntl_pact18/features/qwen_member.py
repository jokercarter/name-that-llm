"""Qwen3-8B member: frozen hidden-state probe or QLoRA fine-tune.

Two modes, same feature path (a forward hook captures one or more decoder
layers and mean-pools them over the attention mask):

* ``probe`` -- weights frozen (4-bit), pooled states -> logistic regression.
  Cheap enough to run per fold, so it keeps the prompt-grouped OOF protocol.
* ``qlora`` -- 4-bit base + LoRA adapters + a linear 18-way head on the last
  token, trained end-to-end. One adapter for the whole training split (a
  5-fold QLoRA is far beyond a laptop budget).

Both require bitsandbytes for 4-bit loading; without it the loader falls back
to fp16 with CPU offload, which is correct but much slower.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..device import DeviceInfo, import_torch
from ..labels import LABELS
from ..utils import normalize_rows


def split_by_group(
    groups: Optional[Sequence[int]], n: int, frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split row indices by prompt group.

    Early stopping must never look at a prompt that also appears in training,
    otherwise the "validation" curve is optimistic and the restored checkpoint
    is chosen on leaked information.
    """
    if groups is None or frac <= 0.0:
        return np.arange(n), np.empty(0, dtype=int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 8:
        return np.arange(n), np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    n_hold = min(max(1, int(round(len(uniq) * frac))), len(uniq) - 1)
    hold = rng.choice(uniq, size=n_hold, replace=False)
    mask = np.isin(groups, hold)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


class QwenMember:
    def __init__(
        self,
        device_info: DeviceInfo,
        mode: str = "probe",
        model_id: str = "Qwen/Qwen3-8B-Base",
        max_len: int = 512,
        batch_size: int = 4,
        layers: Sequence[int] = (18,),
        C: float = 0.0025,
        epochs: int = 1,
        lr: float = 1e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        grad_checkpoint: bool = True,
        load_in_4bit: bool = True,
        supcon_weight: float = 0.1,
        label_smoothing: float = 0.05,
        grad_accum: int = 8,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        time_budget_hours: float = 0.0,
        log_every: int = 20,
        target_modules: Optional[Sequence[str]] = None,
        val_fraction: float = 0.0,
        early_stop_patience: int = 3,
        eval_every: int = 100,
        distributed: bool = True,
        skip_lm_head: bool = True,
        length_bucket: bool = True,
        bucket_pool: int = 40,
        reset_before_fit: bool = True,
        compute_dtype: str = "auto",
        head_proj: int = 512,
        head_dropout: float = 0.1,
        truncate_layers: int = 0,
        pool_mode: str = "attention",
        two_stage: bool = False,
        stage1_epochs: int = 1,
        stage1_lr: float = 1e-4,
        stage2_epochs: int = 2,
        stage2_lr: float = 5e-5,
        head_lr: float = 1e-3,
        stage2_head_lr: float = 5e-4,
        triplet_batch: int = 3,
        centred_weight: float = 1.0,
        stage1_time_budget_hours: float = 0.0,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.device_info = device_info
        self.mode = mode
        self.model_id = model_id
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.layers = tuple(int(v) for v in layers)
        self.C = float(C)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.load_in_4bit = bool(load_in_4bit)
        self.supcon_weight = float(supcon_weight)
        self.label_smoothing = float(label_smoothing)
        self.grad_accum = max(1, int(grad_accum))
        self.warmup_ratio = float(warmup_ratio)
        self.weight_decay = float(weight_decay)
        self.time_budget_hours = float(time_budget_hours)
        self.log_every = max(1, int(log_every))
        self.target_modules = list(target_modules) if target_modules else [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.val_fraction = float(val_fraction)
        self.early_stop_patience = max(1, int(early_stop_patience))
        self.eval_every = max(1, int(eval_every))
        # --- two-GPU data parallelism (no-op when only one process runs) ---
        self.distributed = bool(distributed)
        self.skip_lm_head = bool(skip_lm_head)
        self.length_bucket = bool(length_bucket)
        self.bucket_pool = max(1, int(bucket_pool))
        self.reset_before_fit = bool(reset_before_fit)
        self.compute_dtype = str(compute_dtype).lower()
        self.head_proj = int(head_proj)
        self.head_dropout = float(head_dropout)
        # --- truncated trunk + two-stage training (public 0.59x recipe) ---
        # truncate_layers: keep only the first N decoder blocks. Style lives in
        # the early/middle layers; the last blocks mostly encode topic, and test
        # prompts are unseen, so they are pure compute.
        self.truncate_layers = int(truncate_layers)
        self.pool_mode = str(pool_mode).lower()
        self.two_stage = bool(two_stage)
        self.stage1_epochs = max(1, int(stage1_epochs))
        self.stage1_lr = float(stage1_lr)
        self.stage2_epochs = max(1, int(stage2_epochs))
        self.stage2_lr = float(stage2_lr)
        self.head_lr = float(head_lr)
        self.stage2_head_lr = float(stage2_head_lr)
        self.triplet_batch = max(1, int(triplet_batch))
        self.centred_weight = float(centred_weight)
        self.stage1_time_budget_hours = float(stage1_time_budget_hours)
        self.seed = int(seed)
        self.verbose = verbose

        self.name = f"qwen_{mode}"
        self.model = None
        self.tokenizer = None
        self.head = None          # sklearn LR for probe, torch Linear for qlora
        self.classes_ = None
        self.torch = None
        self.device = None
        self._n_features = 0
        self._cache: Dict[str, np.ndarray] = {}
        # qlora bookkeeping: whether the PEFT adapter is attached, how many
        # fits this member has seen (each fold must start from a fresh adapter),
        # and the lm_head we hide to skip the 151936-way projection.
        self._peft_wrapped = False
        self._fit_count = 0
        self._lm_head = None
        self._qlora_state = None
        self._head_state = None
        # stage-1 adapter (pool-only pretraining) kept on host memory so every
        # fold's stage 2 starts from the same weights instead of from the
        # previous fold's fine-tune
        self._stage1_state = None
        self._stage1_done = False
        # Free the backbone after fit so the next GPU member (XGBoost, ...) gets
        # the whole 8 GB; predict() reloads lazily and reuses the cache.
        self.release_after_fit = True

    # ------------------------------------------------------------------ load
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {msg}", flush=True)

    def _compute_torch_dtype(self, torch):
        """bf16 only where the hardware has native bf16 units (cc >= 8.0).

        ``torch.cuda.is_bf16_supported()`` returns True on a T4 under recent
        torch builds (sm_75 can *emulate* bf16), but the 4-bit kernels then run
        through fp32 emulation, which is both slow and numerically different.
        The compute capability is the authoritative check.
        """
        if self.compute_dtype == "bf16":
            return torch.bfloat16
        if self.compute_dtype == "fp16":
            return torch.float16
        try:
            major, minor = torch.cuda.get_device_capability()
            native = (int(major), int(minor)) >= (8, 0)
        except Exception:  # pragma: no cover - older torch / cpu only
            native = False
        return torch.bfloat16 if native else torch.float16

    def load(self) -> "QwenMember":
        torch = import_torch()
        if torch is None:
            raise RuntimeError("torch is not importable")
        self.torch = torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = self.device_info.torch_device() or torch.device("cpu")
        self._log(f"loading {self.model_id} on {self.device} (4bit={self.load_in_4bit})")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict = {}
        if self.truncate_layers > 0:
            # Keep only the first N decoder blocks. Building the model from a
            # modified config means the upper blocks are never materialised --
            # 8B drops from 36 to 24 blocks, i.e. ~33% less compute per token
            # and ~1.7 GB less 4-bit weight -- instead of merely skipping them
            # at run time.
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_id)
            full = int(getattr(config, "num_hidden_layers", 0) or 0)
            if full and self.truncate_layers < full:
                config.num_hidden_layers = self.truncate_layers
                if getattr(config, "layer_types", None) is not None:
                    config.layer_types = config.layer_types[: self.truncate_layers]
                kwargs["config"] = config
                self._log(
                    f"truncated trunk: {full} -> {self.truncate_layers} blocks "
                    f"(upper blocks never materialised)"
                )
        if self.load_in_4bit and self.device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    # T4 (sm_75) has no bf16 units: requesting bf16 there either
                    # falls back to a slow fp32 emulation or fails inside the
                    # 4-bit kernels. Pick the dtype the card actually supports.
                    bnb_4bit_compute_dtype=self._compute_torch_dtype(torch),
                )
                kwargs["device_map"] = {"": self.device.index or 0}
            except Exception as exc:  # pragma: no cover
                self._log(f"4-bit unavailable ({exc}); falling back to fp16")
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
            kwargs["device_map"] = {"": self.device} if self.device.type == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        if self.device.type == "cuda":
            self.model.config.use_cache = False
        self._log(f"loaded in {time.time() - t0:.1f}s")

        # rebuild the QLoRA adapter + fusion head from the numpy state that
        # __getstate__ stored (the 4-bit base is re-loaded, never serialised)
        if self.mode != "probe" and not self._peft_wrapped and (
            getattr(self, "_qlora_state", None) or getattr(self, "_head_state", None)
        ):
            self._build_qlora()
            lora_state = getattr(self, "_qlora_state", None)
            if lora_state:
                own = dict(self.model.named_parameters())
                with torch.no_grad():
                    for name, array in lora_state.items():
                        if name in own:
                            own[name].copy_(torch.from_numpy(array))
            head_state = getattr(self, "_head_state", None)
            if head_state:
                self.head.load_state_dict(
                    {k: torch.from_numpy(v) for k, v in head_state.items()}
                )
            self._qlora_state = None
            self._head_state = None
        return self

    # -------------------------------------------------------------- features
    def _pooled_hidden(self, texts: Sequence[str]) -> np.ndarray:
        """Mean-pooled hidden states of the configured layers (cached per text).

        The 5-fold OOF loop calls this five times on overlapping subsets; the
        per-text cache makes the backbone run exactly once per passage.
        """
        texts = list(texts)
        missing = [t for t in texts if t not in self._cache]
        if missing:
            if self.model is None:
                self.load()
            fresh = self._encode(missing)
            for text, vector in zip(missing, fresh):
                self._cache[text] = vector
        return np.vstack([self._cache[t] for t in texts])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        captured: Dict[int, "torch.Tensor"] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                captured[layer_idx] = output[0] if isinstance(output, tuple) else output

            return hook

        for layer_idx in self.layers:
            handles.append(self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        feats: List[np.ndarray] = []
        try:
            with torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    batch = list(texts[start:start + self.batch_size])
                    enc = self.tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=self.max_len, return_tensors="pt",
                    ).to(self.device)
                    self.model(**enc)
                    mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
                    pooled = []
                    for layer_idx in self.layers:
                        h = captured[layer_idx].to(torch.float32)
                        pooled.append((h * mask).sum(1) / mask.sum(1).clamp(min=1e-6))
                    feats.append(torch.cat(pooled, dim=1).cpu().numpy())
                    captured.clear()
        finally:
            for handle in handles:
                handle.remove()
        return np.vstack(feats)

    # ------------------------------------------------------------------- API
    def fit(self, texts: Sequence[str], y: Sequence[str],
            groups: Optional[Sequence[int]] = None) -> "QwenMember":
        texts = list(texts)
        self.classes_ = np.unique(np.asarray(list(y)))

        if self.mode == "probe":
            from sklearn.linear_model import LogisticRegression

            t0 = time.time()
            features = self._pooled_hidden(texts)
            self._n_features = features.shape[1]
            self._log(f"features {features.shape} in {time.time() - t0:.1f}s")
            index = {label: i for i, label in enumerate(self.classes_)}
            y_enc = np.asarray([index[v] for v in y], dtype=int)
            self.head = LogisticRegression(C=self.C, max_iter=3000, n_jobs=4)
            self.head.fit(features, y_enc)
            return self

        return self._fit_qlora(texts, y, groups)

    # ------------------------------------------------------------- qlora
    def _ensure_qlora(self) -> None:
        """Load the 4-bit base if needed, then attach a fresh adapter."""
        if self.model is None:
            self.load()
        if not self._peft_wrapped:
            self._build_qlora()

    def _build_qlora(self):
        """4-bit base + LoRA adapters + layer-fusion head (shared by fit/load)."""
        torch = self.torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        from .qlora_head import LayerFusionHead

        # Layer indices are written against one checkpoint (8B has 36 blocks,
        # 1.7B has 28). Clamp instead of crashing when the config is reused.
        n_layers = int(getattr(self.model.config, "num_hidden_layers", 0) or 0)
        if n_layers:
            clipped = tuple(i for i in self.layers if 0 < i <= n_layers)
            if clipped != self.layers:
                self._log(f"layers {list(self.layers)} clipped to "
                          f"{list(clipped)} (model has {n_layers})")
                self.layers = clipped or (n_layers,)

        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=self.grad_checkpoint
        )
        cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )
        self.model = get_peft_model(self.model, cfg)
        hidden = self.model.config.hidden_size
        if self.pool_mode == "mean_concat":
            from .qlora_head import MeanConcatHead

            self.head = MeanConcatHead(
                hidden, len(self.layers), len(LABELS),
                dropout=self.head_dropout,
            ).to(self.device)
        else:
            self.head = LayerFusionHead(
                hidden, len(self.layers), len(LABELS),
                proj=self.head_proj, dropout=self.head_dropout,
            ).to(self.device)
        self._n_features = hidden
        self._peft_wrapped = True
        if self.skip_lm_head:
            self._hide_lm_head()
        self._log(
            f"qlora adapter: r={self.lora_r} on {len(self.target_modules)} projections, "
            f"layers={list(self.layers)}, trainable={self.trainable_parameters():,}"
        )
        return self.head

    def _hide_lm_head(self) -> None:
        """Replace the vocabulary projection with an identity.

        ``CausalLM.forward`` always computes ``logits = lm_head(hidden)``. Those
        logits are never used -- the fusion head produces the 18-way scores --
        so the 151936 x 2048 projection is pure waste: roughly 1.2 GB of
        activation for a batch of 8 at 384 tokens, plus ~6% of forward FLOPs.
        With tied embeddings the projection also duplicates the embedding
        weight, so nothing is lost by skipping it.
        """
        import torch.nn as nn

        target = None
        for module in self.model.modules():
            if hasattr(module, "lm_head") and isinstance(
                getattr(module, "lm_head"), nn.Module
            ):
                target = module
                break
        if target is None or getattr(target.lm_head, "_pact18_hidden", False):
            return
        self._lm_head = target.lm_head
        target.lm_head = nn.Identity()
        target.lm_head._pact18_hidden = True

    def trainable_parameters(self) -> int:
        if self.model is None:
            return 0
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.head is not None:
            n += sum(p.numel() for p in self.head.parameters())
        return int(n)

    def _qlora_logits(self, enc):
        """Return ``(plain_logits, centred_logits_or_None, fused_features)``."""
        out = self.model(**enc, output_hidden_states=True)
        hidden = [out.hidden_states[i] for i in self.layers]
        res = self.head(hidden, enc["attention_mask"])
        if len(res) == 3:
            return res
        logits, fused = res
        return logits, None, fused

    # ------------------------------------------------------------ triplet plan
    def _triplet_units(self, groups, y_idx, rng):
        """Groups -> lists of three rows with three *distinct* labels.

        One passage per label, so a triplet always spans three different models
        answering the same prompt. That is what makes the centred head learn
        "same question, different author" instead of "same topic".
        """
        units = []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            first = {}
            for i in idx:
                first.setdefault(int(y_idx[i]), int(i))
            rows = list(first.values())
            if len(rows) < 3:
                continue
            rng.shuffle(rows)
            for s in range(0, len(rows) - 2, 3):
                units.append(rows[s:s + 3])
        return units

    def _triplet_batches(self, groups, y_idx, world, seed, epoch):
        """Deterministic plan; every rank derives the same batches."""
        rng = np.random.default_rng(seed + 1000 * epoch)
        units = self._triplet_units(groups, y_idx, rng)
        if not units:
            return []
        order = rng.permutation(len(units))
        units = [units[i] for i in order]
        span = self.triplet_batch * max(1, world)
        n = len(units) // span * span
        return [units[s:s + span] for s in range(0, n, span)]

    def _make_batches(self, indices, lengths, global_batch: int, seed: int):
        """Length-bucketed batch plan (kept for the single-stage QLoRA path).

        The two-stage trainer uses :meth:`_triplet_batches` instead, but the
        bucketing helper is still exercised by the unit tests and is the right
        tool when a plain (non-triplet) epoch is needed.
        """
        rng = np.random.default_rng(seed)
        idx = np.asarray(indices)
        if len(idx) < global_batch:
            return []

        if not self.length_bucket:
            order = rng.permutation(idx)
            n = len(order) // global_batch * global_batch
            return [order[s:s + global_batch] for s in range(0, n, global_batch)]

        pool = max(global_batch, global_batch * self.bucket_pool)
        ordered = idx[np.argsort(lengths[idx], kind="stable")]
        batches = []
        carry = np.empty(0, dtype=idx.dtype)
        for s in range(0, len(ordered), pool):
            chunk = np.concatenate([carry, ordered[s:s + pool]])
            n = len(chunk) // global_batch * global_batch
            batches += [chunk[t:t + global_batch] for t in range(0, n, global_batch)]
            carry = chunk[n:]
        if not batches:
            return []
        order = rng.permutation(len(batches))
        return [batches[i] for i in order]

    def _make_optimiser(self, lr: float, head_lr: float):
        torch = self.torch
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters()]
        groups = [
            {"params": lora_params, "lr": lr},
            {"params": head_params, "lr": head_lr},
        ]
        return torch.optim.AdamW(groups, weight_decay=self.weight_decay)

    # --------------------------------------------------------------- training
    def _train(self, texts, y, groups, *, epochs, lr, head_lr,
               deadline=None, stage: str = "stage2", world: int = 1,
               rank: int = 0) -> None:
        """One training run: triplet batches, plain + centred cross-entropy.

        The loss is ``CE(plain) + CE(centred on triplet-centred features)``.
        The second term is the one that makes the representation prompt
        invariant: for three passages about the same volcano, only the residual
        after subtracting their mean still carries the author.
        """
        torch = self.torch
        from .. import dist_utils

        head = self.head
        optimiser = self._make_optimiser(lr, head_lr)
        index = {label: i for i, label in enumerate(LABELS)}
        y_all = np.asarray([index[v] for v in y], dtype=np.int64)

        plan0 = self._triplet_batches(groups, y_all, world, self.seed, 0)
        steps_per_epoch = max(1, len(plan0) // self.grad_accum)
        total_steps = steps_per_epoch * epochs
        warmup = max(1, int(total_steps * self.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
        self.model.train()
        head.train()

        step = micro = 0
        t0 = time.time()
        stopped = False
        for epoch in range(epochs):
            plan = self._triplet_batches(groups, y_all, world, self.seed, epoch)
            if not plan:
                self._log(f"{stage}: no triplet batch (groups too small)")
                break
            for batch in plan:
                my_units = batch[rank * self.triplet_batch:(rank + 1) * self.triplet_batch]
                my = [i for unit in my_units for i in unit]
                enc = self.tokenizer(
                    [texts[i] for i in my], padding=True, truncation=True,
                    max_length=self.max_len, return_tensors="pt",
                ).to(self.device)
                plain, _centred, fused = self._qlora_logits(enc)
                target = torch.as_tensor(y_all[my], device=self.device)
                loss = torch.nn.functional.cross_entropy(
                    plain, target, label_smoothing=self.label_smoothing
                )

                if _centred is not None and len(my_units) and len(my) % 3 == 0:
                    n_trip = len(my) // 3
                    resid = fused.view(n_trip, 3, -1)
                    resid = resid - resid.mean(dim=1, keepdim=True)
                    logits_c = head.centered(resid.reshape(len(my), -1))
                    loss = loss + self.centred_weight * torch.nn.functional.cross_entropy(
                        logits_c, target, label_smoothing=self.label_smoothing
                    )

                if self.supcon_weight > 0 and len(my) > 1:
                    from .qlora_head import supcon_loss

                    loss = loss + self.supcon_weight * supcon_loss(fused, target)

                (loss / self.grad_accum).backward()
                micro += 1
                if micro % self.grad_accum:
                    continue

                if world > 1:
                    dist_utils.all_reduce_grads(
                        [p for g in optimiser.param_groups for p in g["params"]], world
                    )
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimiser.param_groups for p in g["params"]], 1.0
                )
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                step += 1

                if self.log_every and step % self.log_every == 0:
                    over = bool(deadline is not None and time.time() > deadline) if rank == 0 else False
                    if world > 1:
                        over = dist_utils.broadcast_flag(over, self.device)
                    if over:
                        self._log(f"{stage}: time budget reached at step {step}")
                        stopped = True
                        break
                    if rank == 0:
                        lr_now = scheduler.get_last_lr()[0]
                        self._log(
                            f"{stage} ep{epoch + 1} step {step}/{total_steps} "
                            f"loss {float(loss):.4f} lr {lr_now:.2e} "
                            f"({time.time() - t0:.0f}s)"
                        )
            if stopped:
                break

        if micro % self.grad_accum:
            if world > 1:
                dist_utils.all_reduce_grads(
                    [p for g in optimiser.param_groups for p in g["params"]], world
                )
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimiser.param_groups for p in g["params"]], 1.0
            )
            optimiser.step()
            step += 1

        self.model.eval()
        head.eval()
        self._log(f"{stage}: {step} optimizer steps ({micro} micro-batches, "
                  f"{len(plan0)} triplet batches/epoch) in {time.time() - t0:.0f}s")

    def fit_stage1(self, texts: Sequence[str], y: Sequence[str],
                   groups: Optional[Sequence[int]] = None) -> "QwenMember":
        """Pretrain the adapter on the external pool only.

        No host passage is seen here, so this stage cannot leak a label into any
        OOF fold; each fold then starts stage 2 from this same snapshot.
        """
        if not self.two_stage:
            return self
        torch = self.torch or import_torch()
        self.torch = torch
        from .. import dist_utils

        state = dist_utils.current()
        world = state.world if self.distributed else 1
        rank = state.rank if self.distributed else 0

        self._ensure_qlora()
        deadline = (time.time() + self.stage1_time_budget_hours * 3600.0
                    if self.stage1_time_budget_hours > 0 else None)
        self._log(f"stage1: {len(texts)} pool passages -> {self.stage1_epochs} epoch(s)")
        self._train(texts, y, np.asarray(groups if groups is not None else np.arange(len(texts))),
                    epochs=self.stage1_epochs, lr=self.stage1_lr, head_lr=self.head_lr,
                    deadline=deadline, stage="stage1", world=world, rank=rank)

        if world > 1:
            dist_utils.broadcast_tensors(
                [p.data for p in self.model.parameters() if p.requires_grad]
                + [p.data for p in self.head.parameters()]
            )
        # every rank keeps its own snapshot: each one restarts stage 2 from it,
        # and the weights are identical after the broadcast above
        self._stage1_state = {
            "lora": {k: v.detach().float().cpu().numpy()
                     for k, v in self.model.named_parameters() if v.requires_grad},
            "head": {k: v.detach().float().cpu().numpy()
                     for k, v in self.head.state_dict().items()},
        }
        self._stage1_done = True
        self._fit_count = 0
        return self

    def _restore_stage1(self) -> None:
        if not self._stage1_state:
            raise RuntimeError("two_stage fit called before fit_stage1")
        torch = self.torch
        with torch.no_grad():
            own = dict(self.model.named_parameters())
            for key, array in self._stage1_state["lora"].items():
                if key in own:
                    own[key].copy_(torch.from_numpy(array))
            self.head.load_state_dict(
                {k: torch.from_numpy(v) for k, v in self._stage1_state["head"].items()}
            )

    def _fit_qlora(self, texts: Sequence[str], y: Sequence[str],
                   groups: Optional[Sequence[int]] = None) -> "QwenMember":
        torch = self.torch or import_torch()
        self.torch = torch
        from .. import dist_utils

        state = dist_utils.current()
        world = state.world if self.distributed else 1
        rank = state.rank if self.distributed else 0

        if self.two_stage:
            if not self._stage1_state:
                raise RuntimeError(
                    "two_stage member must run fit_stage1() on the pool first"
                )
            self._ensure_qlora()
            if self._fit_count > 0:
                # discard this fold's fine-tune and go back to the pool weights
                self._restore_stage1()
            epochs, lr, head_lr = self.stage2_epochs, self.stage2_lr, self.stage2_head_lr
            stage = "stage2"
        else:
            # Every fold starts from a fresh adapter. Reusing the previous fold's
            # weights would leak that fold's rows into this fold's OOF prediction.
            if self._fit_count > 0 and self.reset_before_fit:
                self.release(capture=False)
                self.head = None
                self._qlora_state = None
                self._head_state = None
            self._ensure_qlora()
            epochs, lr, head_lr = self.epochs, self.lr, self.lr
            stage = "qlora"

        head = self.head
        index = {label: i for i, label in enumerate(LABELS)}
        y_all = np.asarray([index[v] for v in y], dtype=np.int64)
        groups_arr = np.asarray(groups) if groups is not None else np.arange(len(texts))

        deadline = (time.time() + self.time_budget_hours * 3600.0
                    if self.time_budget_hours > 0 else None)
        self._log(
            f"{stage}: train={len(texts)} groups={len(np.unique(groups_arr))} "
            f"world={world} triplets/rank={self.triplet_batch} "
            f"epochs={epochs} lr={lr:g}/{head_lr:g}"
        )
        self._train(texts, y, groups_arr, epochs=epochs, lr=lr, head_lr=head_lr,
                    deadline=deadline, stage=stage, world=world, rank=rank)

        if world > 1:
            dist_utils.broadcast_tensors(
                [p.data for p in self.model.parameters() if p.requires_grad]
                + [p.data for p in head.parameters()]
            )
        self.model.eval()
        head.eval()
        self._fit_count += 1
        return self

    def predict_proba(self, texts: Sequence[str],
                      groups: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.mode == "probe" and self.head is None:
            raise RuntimeError("QwenMember.predict_proba called before fit")
        if self.mode != "probe" and (self.model is None or not self._peft_wrapped):
            self.load()
            self._ensure_qlora()
        texts = list(texts)

        from .. import dist_utils

        state = dist_utils.current()
        if self.distributed and state.world > 1:
            # split the fold's validation rows across cards, then gather: with
            # 50k rows the OOF inference is ~20% of the whole run, so leaving
            # it on one GPU would waste half the hardware
            mine = dist_utils.shard_indices(len(texts), state.rank, state.world)
            local = self._predict_local([texts[i] for i in mine])
            return dist_utils.gather_padded(local, state.world, self.device)
        return self._predict_local(texts)

    def _predict_local(self, texts: Sequence[str]) -> np.ndarray:
        if self.mode == "probe":
            features = self._pooled_hidden(texts)
            raw = self.head.predict_proba(features)
            out = np.full((len(texts), len(LABELS)), 1e-6)
            for col, label in enumerate(self.classes_):
                out[:, LABELS.index(str(label))] = raw[:, col]
            return normalize_rows(out)

        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_len, return_tensors="pt",
                ).to(self.device)
                logits, _, _ = self._qlora_logits(enc)
                out.append(torch.softmax(logits, dim=1).float().cpu().numpy())
        return normalize_rows(np.vstack(out))

    def precompute(self, texts: Sequence[str]) -> None:
        """Extract and cache backbone features for every text in one GPU pass."""
        if self.mode != "probe":
            # QLoRA trains the backbone, so a frozen feature cache would be both
            # useless and enormous (50k x 4096 floats ~ 800 MB)
            return
        self._pooled_hidden(list(texts))

    def save_adapter(self, path: str) -> None:
        if self.mode != "probe" and self.model is not None:
            self.model.save_pretrained(path)
            self._log(f"adapter saved -> {path}")

    def _capture_state(self) -> None:
        """Snapshot the adapter + head as numpy so a later ``load`` restores it."""
        if self.mode == "probe" or self.model is None:
            return
        self._qlora_state = {
            k: v.detach().float().cpu().numpy()
            for k, v in self.model.named_parameters() if v.requires_grad
        }
        if self.head is not None:
            self._head_state = {
                k: v.detach().float().cpu().numpy()
                for k, v in self.head.state_dict().items()
            }

    def release(self, capture: bool = True) -> None:
        """Free the backbone (keeps the fitted head and the feature cache).

        ``capture`` copies the trained adapter to host memory first, so a later
        ``predict_proba`` rebuilds the model and restores those weights instead
        of predicting with a freshly initialised adapter.
        """
        if capture:
            self._capture_state()
        self.model = None
        self.tokenizer = None
        # the adapter died with the backbone: the next fit must re-attach one
        self._peft_wrapped = False
        self._lm_head = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    # ----------------------------------------------------- (de)serialisation
    def __getstate__(self) -> Dict:
        """Drop anything holding a torch/transformers reference (unpickleable
        and/or huge CUDA objects). ``predict`` lazily reloads the backbone via
        :meth:`load`; the fitted ``head`` and numpy feature cache stay.
        """
        state = self.__dict__.copy()
        state["torch"] = None
        state["model"] = None
        state["tokenizer"] = None
        state["device"] = None
        # the per-text cache is regenerated on demand by ``load`` + ``predict``
        state["_cache"] = {}
        # the stage-1 snapshot is only needed while fitting; it is ~100 MB and
        # the final artifact must not carry it
        state["_stage1_state"] = None
        if self.mode != "probe":
            model = self.__dict__.get("model")
            if model is not None:
                state["_qlora_state"] = {
                    k: v.detach().float().cpu().numpy()
                    for k, v in model.named_parameters() if v.requires_grad
                }
            head = self.__dict__.get("head")
            if head is not None:
                state["_head_state"] = {
                    k: v.detach().float().cpu().numpy()
                    for k, v in head.state_dict().items()
                }
            state["head"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        # module refs are restored lazily by load(); keep them None until then
        self.torch = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self._peft_wrapped = False
        self._lm_head = None
        self.__dict__.setdefault("_fit_count", 0)
        self.__dict__.setdefault("_stage1_state", None)
        self.__dict__.setdefault("_stage1_done", False)
        self.__dict__.setdefault("_qlora_state", None)
        self.__dict__.setdefault("_head_state", None)
