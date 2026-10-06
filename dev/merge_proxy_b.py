"""Merge generated proxy-B texts with specs; keep attributes confirmed by blind annotation."""
import json
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent / "proxy_b"
KEYS = ["topic", "sentiment", "form", "tense", "person", "has_number", "has_negation",
        "mentions_person", "urgent", "colloquial", "mentions_place", "long"]
rows, agree = [], Counter()
for gen in sorted(D.glob("gen_*.jsonl")):
    chunk = gen.stem.split("_")[1]
    spec = {r["id"]: r for r in map(json.loads, open(D / f"spec_{chunk}.jsonl", encoding="utf-8"))}
    verp = D / f"ver_{chunk}.jsonl"
    ver = {r["id"]: r for r in map(json.loads, open(verp, encoding="utf-8"))} if verp.exists() else None
    for g in map(json.loads, open(gen, encoding="utf-8")):
        s = spec[g["id"]]
        attrs = {}
        for k in KEYS:
            ok = ver is None or ver.get(g["id"], {}).get(k) == s[k]
            if ver is not None:
                agree[k] += ok
            attrs[k] = s[k] if ok else None
        rows.append({"id": g["id"], "text": g["text"], "attrs": attrs})
n_ver = sum(1 for _ in D.glob("ver_*.jsonl")) * 250
print("rows", len(rows), "agreement:", {k: round(v / n_ver, 2) for k, v in agree.items()})
with open(D / "corpus.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
