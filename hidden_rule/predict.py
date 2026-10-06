"""Generate predictions with a trained episodic adapter (inference only)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from hidden_rule.backbone import Backbone
from hidden_rule.data import read_episodes, write_predictions
from hidden_rule.features import encode_texts, unique_texts
from hidden_rule.model import EpisodeClassifier, joint_decode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dir", type=Path, help="local rubert-tiny2 directory (default: assets/model)")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    model = EpisodeClassifier.from_state_dict(load_file(str(args.adapter), device="cpu"))
    model.eval()

    backbone = Backbone(args.model_dir)
    episodes = read_episodes(args.data)
    support_labels = torch.tensor(
        [[item.label for item in episode.support] for episode in episodes], dtype=torch.long
    )
    groups = [[item.text for item in episode.support + episode.queries] for episode in episodes]
    texts, indices = unique_texts(groups)
    index = torch.tensor(indices, dtype=torch.long)

    with torch.inference_mode():
        encode_started = time.perf_counter()
        features = encode_texts(backbone, texts)
        encode_seconds = time.perf_counter() - encode_started
        probabilities: list[torch.Tensor] = []
        batch_size = max(1, args.batch_size)
        for start in range(0, len(episodes), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(episodes)))
            logits = model(features[index[rows, 8:]], features[index[rows, :8]], support_labels[rows])
            _, marginals = joint_decode(logits, model.count_log_prior)
            probabilities.append(marginals.cpu())
    write_predictions(episodes, torch.cat(probabilities).double(), args.output)
    print(
        json.dumps(
            {
                "episodes": len(episodes),
                "queries": len(episodes) * 4,
                "unique_texts": len(texts),
                "encode_seconds": round(encode_seconds, 2),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
