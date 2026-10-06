"""Trainable episodic heads on cached frozen features (research harness).

Usage: python heads.py TAG CORPUS MODEL [key=value ...]
"""

from __future__ import annotations

import json
import math
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

from harness import joint_decode, load_features, load_split, score


# ----------------------------------------------------------------------------
# feature builders


def build_input(feats: dict, kind: str) -> torch.Tensor:
    if kind == "cls3":
        return feats["cls"][:, 3]
    if kind == "cls+mean3":
        return torch.cat([feats["cls"][:, 3], feats["mean"][:, 3]], 1)
    if kind == "all":
        return torch.cat([feats["cls"].flatten(1), feats["mean"].flatten(1)], 1)
    if kind == "mean_all":
        return feats["mean"].flatten(1)
    if kind == "cls_all":
        return feats["cls"].flatten(1)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# heads: forward(S [E,8,D], y [E,8], Q [E,4,D]) -> logits [E,4]


def prototypes(S, y):
    yf = y.float()[..., None]
    c1 = (S * yf).sum(1) / yf.sum(1).clamp_min(1)
    c0 = (S * (1 - yf)).sum(1) / (1 - yf).sum(1).clamp_min(1)
    return c1, c0


class ProtoHead(nn.Module):
    def __init__(self, D, d=128, metric="cos", hidden=0, dropout=0.1):
        super().__init__()
        if hidden:
            self.enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, d))
        elif d:
            self.enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(D, d))
        else:
            self.enc = nn.Identity()
        self.metric = metric
        self.scale = nn.Parameter(torch.tensor(10.0 if metric == "cos" else 1.0))
        self.bias = nn.Parameter(torch.zeros(()))

    def embed(self, x):
        return self.enc(x)

    def forward(self, S, y, Q):
        S, Q = self.embed(S), self.embed(Q)
        if self.metric == "cos":
            S, Q = F.normalize(S, dim=-1), F.normalize(Q, dim=-1)
        c1, c0 = prototypes(S, y)
        if self.metric == "cos":
            logit = torch.einsum("eqd,ed->eq", Q, F.normalize(c1, dim=-1) - F.normalize(c0, dim=-1))
        else:
            logit = (-((Q - c1[:, None]) ** 2).sum(-1) + ((Q - c0[:, None]) ** 2).sum(-1)) / math.sqrt(Q.shape[-1])
        return self.scale * logit + self.bias


class GatedProtoHead(nn.Module):
    """Projection, then a task-conditioned per-dimension gate from support stats.

    For each dim k: separation t_k = (c1_k - c0_k) / sqrt(pooled within var + eps).
    Gate g = softmax-ish weighting of dims by |t| (learned temperature), applied
    to the difference vector; logit = sum_k g_k * sign... implemented as
    weighted LDA direction w_k = g_k * (c1_k - c0_k) / var_k.
    """

    def __init__(self, D, d=128, dropout=0.1, hidden=0, per_dim=True):
        super().__init__()
        if hidden:
            self.enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, d))
        elif d:
            self.enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(D, d))
        else:
            self.enc = nn.Identity()
            d = D
        self.log_prior_var = nn.Parameter(torch.zeros(d), requires_grad=per_dim)
        self.shrink = nn.Parameter(torch.tensor(0.0))
        self.gate_a = nn.Parameter(torch.tensor(1.0))
        self.gate_b = nn.Parameter(torch.tensor(-1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, S, y, Q):
        S, Q = self.enc(S), self.enc(Q)
        c1, c0 = prototypes(S, y)
        yf = y.float()[..., None]
        resid = S - (yf * c1[:, None] + (1 - yf) * c0[:, None])
        var = (resid ** 2).sum(1) / 6
        lam = torch.sigmoid(self.shrink)
        var = (1 - lam) * var + lam * self.log_prior_var.exp()[None]
        diff = c1 - c0
        t = diff.abs() / (var + 1e-4).sqrt()
        gate = torch.sigmoid(self.gate_a * t + self.gate_b)
        w = gate * diff / (var + 1e-4)
        mid = (c1 + c0) / 2
        logit = torch.einsum("eqd,ed->eq", Q - mid[:, None], w) / math.sqrt(Q.shape[-1])
        return self.scale * logit + self.bias


class SetTransformerHead(nn.Module):
    def __init__(self, D, d=64, layers=2, heads=4, dropout=0.1, ff=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(D, d))
        self.role = nn.Embedding(3, d)  # 0: neg support, 1: pos support, 2: query
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def forward(self, S, y, Q):
        hs = self.inp(S) + self.role(y.long())
        hq = self.inp(Q) + self.role.weight[2][None, None]
        h = self.tr(torch.cat([hs, hq], 1))
        return self.out(h[:, 8:]).squeeze(-1)


class ProtoPlusSet(nn.Module):
    """Proto logit plus a set-transformer correction (residual)."""

    def __init__(self, D, d=96, sd=48):
        super().__init__()
        self.proto = ProtoHead(D, d=d, metric="cos")
        self.set = SetTransformerHead(D, d=sd, layers=1, ff=96)

    def forward(self, S, y, Q):
        return self.proto(S, y, Q) + self.set(S, y, Q)


class DualGated(nn.Module):
    """Gated LDA in a learned space plus gated LDA in the raw space."""

    def __init__(self, D, d=128, dropout=0.1, raw_per_dim=False, raw_scale=1.0, hidden=0):
        super().__init__()
        self.learned = GatedProtoHead(D, d=d, dropout=dropout, hidden=hidden)
        self.raw = GatedProtoHead(D, d=0, per_dim=raw_per_dim)
        with torch.no_grad():
            self.raw.scale.fill_(raw_scale)

    def forward(self, S, y, Q):
        return self.learned(S, y, Q) + self.raw(S, y, Q)


HEADS = {
    "dual": DualGated,
    "proto": ProtoHead,
    "gated": GatedProtoHead,
    "set": SetTransformerHead,
    "protoset": ProtoPlusSet,
}


# ----------------------------------------------------------------------------
# episodic augmentation


def resplit(s_idx, s_y, q_idx, q_y, gen):
    """Re-draw support/query partition from the 12 labelled items of each episode."""
    E = s_idx.shape[0]
    idx = torch.cat([s_idx, q_idx], 1)
    lab = torch.cat([s_y, q_y], 1)
    keys = torch.rand(E, 12, generator=gen)
    # select 4 lowest-key positives and 4 lowest-key negatives as support
    big = 10.0
    kp = keys + big * (1 - lab)  # positives first
    kn = keys + big * lab
    pos_order = kp.argsort(1)[:, :4]
    neg_order = kn.argsort(1)[:, :4]
    sup = torch.cat([pos_order, neg_order], 1)
    mask = torch.ones(E, 12, dtype=torch.bool)
    mask.scatter_(1, sup, False)
    qry = mask.nonzero()[:, 1].reshape(E, 4)
    perm_s = torch.rand(E, 8, generator=gen).argsort(1)
    sup = sup.gather(1, perm_s)
    perm_q = torch.rand(E, 4, generator=gen).argsort(1)
    qry = qry.gather(1, perm_q)
    return idx.gather(1, sup), lab.gather(1, sup), idx.gather(1, qry), lab.gather(1, qry)


# ----------------------------------------------------------------------------
# pseudo-rule episodes built only from unlabeled train texts


class PseudoSampler:
    def __init__(self, X, pool, texts, kinds, gen, n_clusters=24):
        import re

        self.X = X
        self.pool = pool
        self.gen = gen
        self.kinds = kinds
        P = len(pool)
        if "lex" in kinds:
            words = [set(w for w in re.findall(r"\w+", texts[i].lower()) if len(w) >= 3) for i in pool.tolist()]
            df: dict[str, list[int]] = {}
            for j, ws in enumerate(words):
                for w in ws:
                    df.setdefault(w, []).append(j)
            self.lex = [torch.tensor(v) for w, v in df.items() if 0.01 * P <= len(v) <= 0.5 * P and len(v) >= 8]
            self.lex_mask = []
            for v in self.lex:
                m = torch.zeros(P, dtype=torch.bool)
                m[v] = True
                self.lex_mask.append(m)
        if "clu" in kinds:
            Xp = F.normalize(X[pool], dim=-1)
            cent = Xp[torch.randperm(P, generator=gen)[:n_clusters]]
            for _ in range(15):
                assign = (Xp @ cent.T).argmax(1)
                for c in range(n_clusters):
                    sel = assign == c
                    if sel.any():
                        cent[c] = F.normalize(Xp[sel].mean(0), dim=0)
            self.assign = assign
            self.n_clusters = n_clusters

    def _episode(self, pos_mask):
        P = len(self.pool)
        pos = pos_mask.nonzero().squeeze(1)
        neg = (~pos_mask).nonzero().squeeze(1)
        k = int((torch.rand(4, generator=self.gen) < 0.5).sum())
        if len(pos) < 4 + k or len(neg) < 8 - k:
            return None
        pi = pos[torch.randperm(len(pos), generator=self.gen)[: 4 + k]]
        ni = neg[torch.randperm(len(neg), generator=self.gen)[: 8 - k]]
        s = torch.cat([pi[:4], ni[:4]])
        sy = torch.tensor([1] * 4 + [0] * 4)
        q = torch.cat([pi[4:], ni[4:]])
        qy = torch.tensor([1] * k + [0] * (4 - k))
        ps = torch.randperm(8, generator=self.gen)
        pq = torch.randperm(4, generator=self.gen)
        return self.pool[s[ps]], sy[ps], self.pool[q[pq]], qy[pq]

    def sample(self, n):
        out = []
        while len(out) < n:
            kind = self.kinds[int(torch.randint(0, len(self.kinds), (1,), generator=self.gen))]
            if kind == "dir":
                v = F.normalize(torch.randn(self.X.shape[1], generator=self.gen), dim=0)
                p = self.X[self.pool] @ v
                lo, hi = p.quantile(0.35), p.quantile(0.65)
                mask = p > hi
                # restrict negatives to the bottom tail
                ep = self._episode_tails(mask, p < lo)
            elif kind == "lex":
                j = int(torch.randint(0, len(self.lex), (1,), generator=self.gen))
                ep = self._episode(self.lex_mask[j])
            else:
                c = int(torch.randint(0, self.n_clusters, (1,), generator=self.gen))
                ep = self._episode(self.assign == c)
            if ep is not None:
                out.append(ep)
        return [torch.stack(t) for t in zip(*out)]

    def _episode_tails(self, pos_mask, neg_mask):
        pos = pos_mask.nonzero().squeeze(1)
        neg = neg_mask.nonzero().squeeze(1)
        k = int((torch.rand(4, generator=self.gen) < 0.5).sum())
        pi = pos[torch.randperm(len(pos), generator=self.gen)[: 4 + k]]
        ni = neg[torch.randperm(len(neg), generator=self.gen)[: 8 - k]]
        s = torch.cat([pi[:4], ni[:4]])
        sy = torch.tensor([1] * 4 + [0] * 4)
        q = torch.cat([pi[4:], ni[4:]])
        qy = torch.tensor([1] * k + [0] * (4 - k))
        ps = torch.randperm(8, generator=self.gen)
        pq = torch.randperm(4, generator=self.gen)
        return self.pool[s[ps]], sy[ps], self.pool[q[pq]], qy[pq]


# ----------------------------------------------------------------------------


def run(tag, corpus, model_name, **cfg):
    torch.manual_seed(int(cfg.get("seed", 0)))
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    feats = load_features(corpus)
    X = build_input(feats, cfg.get("feats", "cls+mean3"))
    train = load_split(tag, "train")
    stats_idx = torch.unique(torch.cat([train.s_idx.flatten(), train.q_idx.flatten()]))
    mu, sd = X[stats_idx].mean(0), X[stats_idx].std(0) + 1e-4
    X = (X - mu) / sd
    D = X.shape[1]
    kwargs = {k: v for k, v in cfg.items() if k in {"d", "metric", "hidden", "dropout", "layers", "heads", "ff", "sd", "per_dim", "raw_per_dim", "raw_scale"}}
    model = HEADS[model_name](D, **kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    steps = int(cfg.get("steps", 3000))
    bs = int(cfg.get("bs", 64))
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("wd", 0.01))
    flip = float(cfg.get("flip", 0.5))
    aug = int(cfg.get("resplit", 1))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    E = train.s_idx.shape[0]
    counts = train.q_y.sum(1)
    log_prior = torch.log(torch.bincount(counts, minlength=5).float() + 1) - math.log(E + 5)
    pseudo = cfg.get("pseudo", "")
    pmix = float(cfg.get("pmix", 0.5))
    sampler = None
    if pseudo:
        from episodes import load_corpus

        texts = [row["text"] for row in load_corpus(corpus)]
        sampler = PseudoSampler(X, stats_idx, texts, pseudo.split("+"), gen)
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        b = torch.randint(0, E, (bs,), generator=gen)
        s_idx, s_y, q_idx, q_y = train.s_idx[b], train.s_y[b], train.q_idx[b], train.q_y[b]
        if aug:
            s_idx, s_y, q_idx, q_y = resplit(s_idx, s_y, q_idx, q_y, gen)
        if sampler is not None:
            n_ps = int(round(pmix * bs))
            if n_ps:
                ps = sampler.sample(n_ps)
                s_idx = torch.cat([s_idx[: bs - n_ps], ps[0]])
                s_y = torch.cat([s_y[: bs - n_ps], ps[1]])
                q_idx = torch.cat([q_idx[: bs - n_ps], ps[2]])
                q_y = torch.cat([q_y[: bs - n_ps], ps[3]])
        if flip:
            f = (torch.rand(bs, 1, generator=gen) < flip).long()
            s_y = s_y ^ f
            q_y = q_y ^ f
        logits = model(X[s_idx], s_y, X[q_idx])
        loss = F.binary_cross_entropy_with_logits(logits, q_y.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    train_s = time.perf_counter() - t0
    model.eval()
    out = {"model": model_name, "params": n_params, "cfg": cfg, "train_s": round(train_s, 1)}
    with torch.no_grad():
        for name in ["valid", "test_seen", "test_unseen"]:
            split = load_split(tag, name)
            logits = model(X[split.s_idx], split.s_y, X[split.q_idx])
            r = score(split, logits)
            rj = score(split, joint_decode(logits, log_prior))
            out[name] = {"em": round(r["em"], 3), "acc": round(r["acc"], 3), "em_joint": round(rj["em"], 3)}
            if cfg.get("families"):
                out[name]["fam"] = r["by_family"]
    return out


if __name__ == "__main__":
    tag, corpus, model_name = sys.argv[1:4]
    cfg = {}
    for kv in sys.argv[4:]:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    print(json.dumps(run(tag, corpus, model_name, **cfg), ensure_ascii=False))
