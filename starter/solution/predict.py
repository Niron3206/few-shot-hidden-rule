"""Generate predictions with the trained episodic adapter (inference only)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from starter.runtime.backbone import FrozenBackbone
from starter.runtime.meta import validate_artifact, verify_backbone_sha
from starter.solution.data import (
    iter_episode_batches,
    prediction_rows,
    read_episodes,
    write_jsonl,
)
from starter.solution.features import encode_texts, unique_texts
from starter.solution.model import ParticipantModule, joint_decode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(2)
    verify_backbone_sha()
    validate_artifact(args.adapter)
    model = ParticipantModule.from_state_dict(load_file(str(args.adapter), device="cpu"))
    model.eval()

    backbone = FrozenBackbone()
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
        for rows in iter_episode_batches(len(episodes), max(1, args.batch_size), shuffle=False):
            logits = model(
                features[index[rows, 8:]],
                features[index[rows, :8]],
                support_labels[rows],
            )
            _, marginals = joint_decode(logits, model.count_log_prior)
            probabilities.append(marginals.cpu())
    output_probabilities = torch.cat(probabilities).double()
    write_jsonl(prediction_rows(episodes, output_probabilities), args.output)
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
