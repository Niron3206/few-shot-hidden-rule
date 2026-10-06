"""Strictly validate a predictions file and report Episode Exact Match.

Usage: python dev/evaluate.py DATA.jsonl LABELS.jsonl PREDICTIONS.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def main(data_path: str, labels_path: str, pred_path: str) -> None:
    expected = {}
    for line in open(data_path, encoding="utf-8"):
        if line.strip():
            ep = json.loads(line)
            for q in ep["queries"]:
                expected[(ep["episode_id"], q["query_id"])] = None
    truth, family = {}, {}
    for line in open(labels_path, encoding="utf-8"):
        row = json.loads(line)
        key = (row["episode_id"], row["query_id"])
        truth[key] = row["label"]
        family[row["episode_id"]] = row.get("family", "?")
    seen = set()
    correct = defaultdict(list)
    for number, line in enumerate(open(pred_path, encoding="utf-8"), 1):
        row = json.loads(line)
        assert set(row) == {"episode_id", "query_id", "label", "probability"}, f"line {number}: fields"
        key = (row["episode_id"], row["query_id"])
        assert key in expected, f"line {number}: unknown id"
        assert key not in seen, f"line {number}: duplicate"
        seen.add(key)
        label, prob = row["label"], row["probability"]
        assert isinstance(label, int) and not isinstance(label, bool) and label in (0, 1), f"line {number}: label"
        assert isinstance(prob, (int, float)) and math.isfinite(prob) and 0 <= prob <= 1, f"line {number}: prob"
        assert label == int(prob >= 0.5), f"line {number}: label/probability mismatch"
        correct[row["episode_id"]].append(label == truth[key])
    assert seen == set(expected), "missing predictions"
    em = {ep: all(v) for ep, v in correct.items()}
    by_family = defaultdict(list)
    for ep, ok in em.items():
        by_family[family[ep]].append(ok)
    print(json.dumps({
        "episode_exact_match": round(sum(em.values()) / len(em), 4),
        "query_accuracy": round(sum(sum(v) for v in correct.values()) / len(truth), 4),
        "by_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_family.items())},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
