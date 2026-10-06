"""Non-parametric few-shot baselines on cached frozen features."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from harness import load_features, load_split, score


def build_feats(feats: dict, kind: str, stats_idx: torch.Tensor) -> torch.Tensor:
    if kind == "cls3":
        x = feats["cls"][:, 3]
    elif kind == "mean3":
        x = feats["mean"][:, 3]
    elif kind == "cls+mean3":
        x = torch.cat([feats["cls"][:, 3], feats["mean"][:, 3]], 1)
    elif kind == "all":
        x = torch.cat([feats["cls"].flatten(1), feats["mean"].flatten(1)], 1)
    elif kind == "mean_all":
        x = feats["mean"].flatten(1)
    else:
        raise ValueError(kind)
    mu = x[stats_idx].mean(0)
    sd = x[stats_idx].std(0) + 1e-4
    return (x - mu) / sd


def proto_cos(x, split):
    S = F.normalize(x[split.s_idx], dim=-1)
    Q = F.normalize(x[split.q_idx], dim=-1)
    y = split.s_y.float()
    c1 = (S * y[..., None]).sum(1) / 4
    c0 = (S * (1 - y[..., None])).sum(1) / 4
    return torch.einsum("eqd,ed->eq", Q, F.normalize(c1, dim=-1) ) - torch.einsum("eqd,ed->eq", Q, F.normalize(c0, dim=-1))


def proto_euc(x, split):
    S, Q = x[split.s_idx], x[split.q_idx]
    y = split.s_y.float()
    c1 = (S * y[..., None]).sum(1) / 4
    c0 = (S * (1 - y[..., None])).sum(1) / 4
    return -((Q - c1[:, None]) ** 2).sum(-1) + ((Q - c0[:, None]) ** 2).sum(-1)


def diag_lda(x, split, shrink=1.0):
    S, Q = x[split.s_idx], x[split.q_idx]
    y = split.s_y.float()[..., None]
    c1 = (S * y).sum(1) / 4
    c0 = (S * (1 - y)).sum(1) / 4
    resid = S - (y * c1[:, None] + (1 - y) * c0[:, None])
    var = (resid ** 2).sum(1) / 6  # [E, D]
    var = (var * 6 + shrink * 1.0 * 6) / (6 + shrink * 6)
    w = (c1 - c0) / var
    mid = (c1 + c0) / 2
    return torch.einsum("eqd,ed->eq", Q - mid[:, None], w)


def knn_vote(x, split, tau=10.0):
    S = F.normalize(x[split.s_idx], dim=-1)
    Q = F.normalize(x[split.q_idx], dim=-1)
    sim = torch.einsum("eqd,esd->eqs", Q, S)
    sign = (2 * split.s_y.float() - 1)[:, None]
    att = torch.softmax(tau * sim, -1)
    return (att * sign).sum(-1)


def main(tag: str, corpus: str) -> None:
    feats = load_features(corpus)
    train = load_split(tag, "train")
    stats_idx = torch.unique(torch.cat([train.s_idx.flatten(), train.q_idx.flatten()]))
    for name in ["test_seen", "test_unseen"]:
        split = load_split(tag, name)
        print(f"== {tag} {name}")
        for kind in ["cls3", "mean3", "cls+mean3", "mean_all", "all"]:
            x = build_feats(feats, kind, stats_idx)
            rows = []
            for method, fn in [("proto_cos", proto_cos), ("proto_euc", proto_euc), ("knn", knn_vote), ("dlda", diag_lda)]:
                r = score(split, fn(x, split))
                rows.append(f"{method}={r['em']:.3f}/{r['acc']:.3f}")
            print(f"  {kind:10s} " + "  ".join(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
