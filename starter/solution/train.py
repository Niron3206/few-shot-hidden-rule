"""Train the episodic few-shot adapter entirely offline on CPU.

Every distinct train text is encoded once by the frozen backbone (CLS and mean
states of all layers).  The adapter is then meta-trained on episodes:

* each step draws real train episodes and re-partitions their twelve labelled
  messages into a fresh 4+4 support set and four queries;
* labels are flipped for half of the episodes so the head stays symmetric;
* a fraction of episodes are lexical pseudo-rules ("contains word w") built
  from the train texts, which keeps the projection from discarding features
  that are irrelevant to the train rules but may matter for unseen ones.

``--epochs`` and ``--batch-size`` mean passes over the train episodes and
episodes per optimizer step, but the schedule is bounded
below so a short command still produces a converged adapter, and bounded above
by step count and wall time so training fits the time limit.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from starter.runtime.backbone import FrozenBackbone
from starter.runtime.meta import validate_artifact, validate_trainable_parameter_count
from starter.solution.data import Episode, read_episodes
from starter.solution.features import encode_texts, unique_texts
from starter.solution.model import ParticipantModule

MIN_BATCH_EPISODES = 64
MIN_STEPS = 6000
MAX_STEPS = 12000
MAX_TRAIN_SECONDS = 12 * 60
PSEUDO_FRACTION = 0.2
FLIP_PROBABILITY = 0.5
WEIGHT_DECAY = 0.01


def episode_tensors(episodes: Sequence[Episode]) -> tuple[list[str], dict[str, torch.Tensor]]:
    groups = [[item.text for item in episode.support + episode.queries] for episode in episodes]
    texts, indices = unique_texts(groups)
    index = torch.tensor(indices, dtype=torch.long)
    labels = torch.tensor(
        [[item.label for item in episode.support + episode.queries] for episode in episodes],
        dtype=torch.long,
    )
    return texts, {"index": index, "labels": labels}


def resplit(index: torch.Tensor, labels: torch.Tensor, generator: torch.Generator):
    """Draw a new 4+4 support and the remaining 4 queries from 12 labelled items."""

    batch = index.shape[0]
    keys = torch.rand(batch, 12, generator=generator)
    positive_first = (keys + 10.0 * (1 - labels)).argsort(1)[:, :4]
    negative_first = (keys + 10.0 * labels).argsort(1)[:, :4]
    support = torch.cat((positive_first, negative_first), 1)
    rest = torch.ones(batch, 12, dtype=torch.bool)
    rest.scatter_(1, support, False)
    queries = rest.nonzero()[:, 1].reshape(batch, 4)
    support = support.gather(1, torch.rand(batch, 8, generator=generator).argsort(1))
    queries = queries.gather(1, torch.rand(batch, 4, generator=generator).argsort(1))
    return (
        index.gather(1, support),
        labels.gather(1, support),
        index.gather(1, queries),
        labels.gather(1, queries),
    )


class LexicalRules:
    """Pseudo-episodes whose rule is the presence of one frequent word."""

    def __init__(self, texts: Sequence[str], generator: torch.Generator) -> None:
        self.generator = generator
        count = len(texts)
        postings: dict[str, list[int]] = {}
        for position, text in enumerate(texts):
            for word in set(re.findall(r"\w+", text.lower())):
                if len(word) >= 3:
                    postings.setdefault(word, []).append(position)
        low, high = max(8, int(0.01 * count)), int(0.5 * count)
        self.masks = []
        for word in sorted(postings):
            members = postings[word]
            if low <= len(members) <= high and count - len(members) >= 8:
                mask = torch.zeros(count, dtype=torch.bool)
                mask[members] = True
                self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.masks)

    def sample(self, size: int):
        rows = []
        for _ in range(size):
            mask = self.masks[int(torch.randint(len(self.masks), (1,), generator=self.generator))]
            positives = mask.nonzero().squeeze(1)
            negatives = (~mask).nonzero().squeeze(1)
            k = int((torch.rand(4, generator=self.generator) < 0.5).sum())
            pos = positives[torch.randperm(len(positives), generator=self.generator)[: 4 + k]]
            neg = negatives[torch.randperm(len(negatives), generator=self.generator)[: 8 - k]]
            support = torch.cat((pos[:4], neg[:4]))
            support_labels = torch.tensor([1] * 4 + [0] * 4)
            queries = torch.cat((pos[4:], neg[4:]))
            query_labels = torch.tensor([1] * k + [0] * (4 - k))
            order_s = torch.randperm(8, generator=self.generator)
            order_q = torch.randperm(4, generator=self.generator)
            rows.append((support[order_s], support_labels[order_s], queries[order_q], query_labels[order_q]))
        return [torch.stack(column) for column in zip(*rows)]


def count_log_prior(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels[:, 8:].sum(1), minlength=5).double()
    return torch.log((counts + 0.5) / (counts.sum() + 2.5)).float()


def save_adapter(model: ParticipantModule, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(output),
    )
    validate_artifact(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-episodes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("learning-rate must be a positive finite number")
    started = time.perf_counter()
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    backbone = FrozenBackbone()
    episodes = read_episodes(args.train, require_query_labels=True)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    texts, tensors = episode_tensors(episodes)
    index, labels = tensors["index"], tensors["labels"]

    encode_started = time.perf_counter()
    features = encode_texts(backbone, texts)
    encode_seconds = time.perf_counter() - encode_started
    del backbone

    model = ParticipantModule(input_dim=features.shape[1])
    model.set_feature_stats(features)
    model.count_log_prior.copy_(count_log_prior(labels))
    trainable = validate_trainable_parameter_count(model)

    lexical = LexicalRules(texts, generator)
    episode_count = len(episodes)
    batch_episodes = max(args.batch_size, MIN_BATCH_EPISODES)
    requested = args.epochs * math.ceil(episode_count / batch_episodes)
    steps = min(MAX_STEPS, max(requested, min(MIN_STEPS, 50 * episode_count)))
    pseudo = int(round(PSEUDO_FRACTION * batch_episodes)) if len(lexical) else 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=steps, pct_start=0.1
    )
    train_started = time.perf_counter()
    losses: list[float] = []
    running, seen, done = 0.0, 0, 0
    model.train()
    for step in range(steps):
        rows = torch.randint(episode_count, (batch_episodes - pseudo,), generator=generator)
        s_idx, s_y, q_idx, q_y = resplit(index[rows], labels[rows], generator)
        if pseudo:
            p_s, p_sy, p_q, p_qy = lexical.sample(pseudo)
            s_idx, s_y = torch.cat((s_idx, p_s)), torch.cat((s_y, p_sy))
            q_idx, q_y = torch.cat((q_idx, p_q)), torch.cat((q_y, p_qy))
        flip = (torch.rand(len(s_y), 1, generator=generator) < FLIP_PROBABILITY).long()
        s_y, q_y = s_y ^ flip, q_y ^ flip
        logits = model(features[q_idx], features[s_idx], s_y)
        loss = F.binary_cross_entropy_with_logits(logits, q_y.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        running += float(loss.detach()) * logits.numel()
        seen += logits.numel()
        done = step + 1
        if done % 500 == 0 or done == steps:
            losses.append(running / seen)
            running, seen = 0.0, 0
            if time.perf_counter() - train_started > MAX_TRAIN_SECONDS:
                break
    train_seconds = time.perf_counter() - train_started

    model.eval()
    save_adapter(model, args.adapter)
    artifact = validate_artifact(args.adapter)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "episodes": episode_count,
                "unique_texts": len(texts),
                "steps": done,
                "planned_steps": steps,
                "batch_episodes": batch_episodes,
                "pseudo_rules": len(lexical),
                "losses": losses,
                "encode_seconds": round(encode_seconds, 2),
                "train_seconds": round(train_seconds, 2),
                "total_seconds": round(time.perf_counter() - started, 2),
                "trainable_parameters": trainable,
                "adapter_bytes": artifact.size_bytes,
                "adapter_scalars": artifact.scalar_count,
                "query_count_prior": model.count_log_prior.exp().tolist(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
