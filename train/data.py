"""Packed-sequence dataloader over token, loss-mask, and target-weight shards.

Legacy corpora may pair the flat uint16 token stream with a binary uint8 loss
mask (1 = loss ON). Relational corpora instead use per-arm target-weight
sidecars and normally omit the legacy mask. Batches are contiguous windows;
when a legacy mask is present, the target at position t is token t+1 and its
label is -100 wherever the NEXT token's mask is 0.

The cursor is a single integer (token offset), saved into checkpoints so a
resumed run continues on the exact next batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PackedShards:
    def __init__(
        self,
        bin_path: str | Path,
        mask_path: str | Path | None,
        ctx: int,
        batch_size: int,
        device: str = "cpu",
        start_cursor: int = 0,
        seed: int = 0,
        weights_path: str | Path | None = None,
    ):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if mask_path is not None and Path(mask_path).exists():
            self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
            assert len(self.mask) == len(self.tokens), "mask/token length mismatch"
        else:
            self.mask = None
        if weights_path is not None:
            self.target_weights = np.memmap(weights_path, dtype=np.uint8, mode="r")
            assert len(self.target_weights) == len(
                self.tokens
            ), "weights/token length mismatch"
        else:
            self.target_weights = None
        self.ctx = ctx
        self.batch_size = batch_size
        self.device = device
        self.cursor = start_cursor
        self.n_tokens = len(self.tokens)
        self.epoch = 0
        span = self.batch_size * (self.ctx + 1)
        assert self.n_tokens > span, "corpus smaller than one batch"

    def _window(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray | None]:
        toks = np.asarray(self.tokens[start : start + length])
        msk = np.asarray(self.mask[start : start + length]) if self.mask is not None else None
        return toks, msk

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        span = self.batch_size * (self.ctx + 1)
        if self.cursor + span >= self.n_tokens:
            self.cursor = 0
            self.epoch += 1
        toks, msk = self._window(self.cursor, span)
        self.cursor += self.batch_size * self.ctx  # overlap of 1 keeps every target trained
        toks = toks.astype(np.int64).reshape(self.batch_size, self.ctx + 1)
        x = torch.from_numpy(toks[:, :-1].copy())
        y = torch.from_numpy(toks[:, 1:].copy())
        if msk is not None:
            m = msk.reshape(self.batch_size, self.ctx + 1)[:, 1:]
            y[torch.from_numpy((m == 0).copy())] = -100
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def _aligned_next_token_weights_for_last_batch(self) -> torch.Tensor:
        assert self.target_weights is not None
        span = self.batch_size * (self.ctx + 1)
        start = self.cursor - self.batch_size * self.ctx
        raw = np.asarray(self.target_weights[start : start + span])
        raw = raw.reshape(self.batch_size, self.ctx + 1)[:, 1:]
        weights = torch.from_numpy(raw.astype(np.float32, copy=True))
        if self.device == "cuda":
            weights = weights.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            weights = weights.to(self.device)
        return weights

    def next_weighted_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, targets = self.next_batch()
        if self.target_weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
        else:
            weights = self._aligned_next_token_weights_for_last_batch()
        return x, targets, weights

    def masked_value_batch(self, max_batches: int = 8) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Probe positions excluded by an optional legacy binary loss mask.

        Returns (x, y) where y is -100 everywhere EXCEPT masked-value targets —
        the complement of the training labels — sampled from the shard head.
        Returns None for target-weight-only relational corpora and for legacy
        corpora with no masked positions.
        """
        if self.mask is None:
            return None
        span = self.batch_size * (self.ctx + 1)
        toks, msk = self._window(0, span * max_batches)
        if (msk == 0).sum() == 0:
            return None
        usable = (len(toks) // (self.ctx + 1)) * (self.ctx + 1)
        toks = toks[:usable].astype(np.int64).reshape(-1, self.ctx + 1)
        msk = msk[:usable].reshape(-1, self.ctx + 1)
        x = torch.from_numpy(toks[:, :-1].copy())
        y = torch.from_numpy(toks[:, 1:].copy())
        keep = torch.from_numpy((msk[:, 1:] == 0).copy())
        y[~keep] = -100
        rows = keep.any(dim=1)
        if not rows.any():
            return None
        return x[rows], y[rows]

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.cursor = state["cursor"]
        self.epoch = state.get("epoch", 0)
