"""Shared research harness: load cached features and episodic index splits."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

DEV = Path(__file__).resolve().parent
ROOT = DEV.parent
sys.path.insert(0, str(DEV))

import features as feature_cache  # noqa: E402


@dataclass
class Split:
    s_idx: torch.Tensor  # [E, 8]
    s_y: torch.Tensor  # [E, 8]
    q_idx: torch.Tensor  # [E, 4]
    q_y: torch.Tensor  # [E, 4]
    family: list[str]
    rule: list[str]


def load_split(tag: str, name: str) -> Split:
    path = ROOT / "runs" / "proxy" / tag / f"{name}.index.json"
    episodes = json.loads(path.read_text(encoding="utf-8"))
    return Split(
        s_idx=torch.tensor([[s["idx"] for s in ep["support"]] for ep in episodes]),
        s_y=torch.tensor([[s["label"] for s in ep["support"]] for ep in episodes]),
        q_idx=torch.tensor([[q["idx"] for q in ep["queries"]] for ep in episodes]),
        q_y=torch.tensor([[q["label"] for q in ep["queries"]] for ep in episodes]),
        family=[ep["family"] for ep in episodes],
        rule=[ep["rule"] for ep in episodes],
    )


def load_features(corpus: str) -> dict:
    return feature_cache.load(corpus)


def score(split: Split, logits: torch.Tensor) -> dict:
    """logits [E, 4] (>0 means label 1) or hard labels via ``pred``."""
    pred = (logits > 0).long() if logits.is_floating_point() else logits
    correct = pred == split.q_y
    em = correct.all(1).float()
    by_family: dict[str, list[float]] = defaultdict(list)
    for fam, value in zip(split.family, em.tolist()):
        by_family[fam].append(value)
    return {
        "em": float(em.mean()),
        "acc": float(correct.float().mean()),
        "by_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_family.items())},
    }


def joint_decode(logits: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
    """MAP labeling over 16 configurations with a prior over the positive count.

    Per-query posteriors already contain an iid Bernoulli(0.5) prior, so the
    count prior enters relative to the binomial: log p(k) - log C(4, k).
    """
    configs = torch.tensor([[(m >> k) & 1 for k in range(4)] for m in range(16)], dtype=torch.float32)
    logp1 = torch.nn.functional.logsigmoid(logits)
    logp0 = torch.nn.functional.logsigmoid(-logits)
    joint = logp1 @ configs.T + logp0 @ (1 - configs).T  # [E, 16]
    binom = torch.log(torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]))
    joint = joint + (log_prior - binom)[configs.sum(1).long()][None]
    return configs[joint.argmax(1)].long()
