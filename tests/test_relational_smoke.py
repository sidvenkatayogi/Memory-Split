from __future__ import annotations

import json

import torch

from scripts.relational_smoke_test import run_smoke


def test_local_pipeline_uses_real_paired_training_and_evaluation(tmp_path):
    report = run_smoke(tmp_path, steps=2, device="cpu")

    assert report == {
        "shared_stream": True,
        "dense_steps": 2,
        "split_steps": 2,
        "resume_exact": True,
        "memory_modes": ["off", "on"],
        "pairs_complete": True,
    }
    assert json.loads((tmp_path / "smoke-report.json").read_text()) == report

    for arm in ("dense", "split"):
        checkpoint = torch.load(
            tmp_path / "runs" / arm / "ckpt.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["step"] == 2
        config = checkpoint["cfg"]
        assert config["train_bin"] == str(tmp_path / "corpus" / "train.bin")
        assert config["train_weights"] == str(
            tmp_path / "corpus" / f"{arm}.weights.bin"
        )

    for mode in ("off", "on"):
        summary = json.loads(
            (tmp_path / "evals" / f"memory_{mode}" / "summary.json").read_text()
        )
        assert summary["memory"] == mode
        assert summary["n_pairs_per_task"] == 4
        assert all(
            task["n_pairs"] == 4 and task["n_rows"] == 8
            for task in summary["tasks"].values()
        )
