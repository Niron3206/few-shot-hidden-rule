"""Cache frozen rubert-tiny2 features for every proxy corpus text.

Stored per text (indexed like the corpus file):
  cls   [N, 4, 312]  CLS state of embeddings output and each of 3 layers
  mean  [N, 4, 312]  masked mean over all non-pad tokens, same layers
  tok   [N, T, 312]  last-layer token states (fp16), T = longest sequence
  tok0  [N, T, 312]  embedding-layer token states (fp16)
  mask  [N, T]       attention mask
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

DEV = Path(__file__).resolve().parent
ROOT = DEV.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DEV))

from hidden_rule.backbone import Backbone  # noqa: E402
from episodes import load_corpus  # noqa: E402


def cache_path(corpus: str) -> Path:
    return ROOT / "runs" / "dev_cache" / f"{corpus}.pt"


@torch.no_grad()
def build(corpus: str) -> dict:
    torch.set_num_threads(2)
    rows = load_corpus(corpus)
    texts = [row["text"] for row in rows]
    backbone = Backbone()
    encoded = backbone.tokenizer(texts, truncation=True, max_length=58)
    lengths = torch.tensor([len(ids) for ids in encoded["input_ids"]])
    T = int(lengths.max())
    N = len(texts)
    cls = torch.zeros(N, 4, 312)
    mean = torch.zeros(N, 4, 312)
    tok = torch.zeros(N, T, 312, dtype=torch.float16)
    tok0 = torch.zeros(N, T, 312, dtype=torch.float16)
    mask = torch.zeros(N, T, dtype=torch.long)
    order = torch.argsort(lengths)
    started = time.perf_counter()
    for start in range(0, N, 128):
        idx = order[start : start + 128]
        L = int(lengths[idx].max())
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        am = torch.zeros(len(idx), L, dtype=torch.long)
        for row, i in enumerate(idx.tolist()):
            seq = encoded["input_ids"][i]
            ids[row, : len(seq)] = torch.tensor(seq)
            am[row, : len(seq)] = 1
        out = backbone.model(input_ids=ids, attention_mask=am, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 1)  # [B, 4, L, H]
        m = am[:, None, :, None].float()
        cls[idx] = hs[:, :, 0]
        mean[idx] = (hs * m).sum(2) / m.sum(2)
        tok[idx, :L] = hs[:, -1].half()
        tok0[idx, :L] = hs[:, 0].half()
        mask[idx, :L] = am
    print(f"{corpus}: {N} texts, T={T}, {time.perf_counter() - started:.1f}s")
    data = {"cls": cls, "mean": mean, "tok": tok, "tok0": tok0, "mask": mask, "lengths": lengths}
    path = cache_path(corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def load(corpus: str) -> dict:
    path = cache_path(corpus)
    if not path.exists():
        return build(corpus)
    return torch.load(path)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        build(name)
