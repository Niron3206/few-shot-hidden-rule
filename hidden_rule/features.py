"""Frozen multi-layer sentence features shared by training and prediction."""

from __future__ import annotations

from typing import Sequence

import torch

from hidden_rule.backbone import Backbone

MAX_LENGTH = 58


@torch.no_grad()
def encode_texts(
    backbone: Backbone,
    texts: Sequence[str],
    *,
    batch_size: int = 128,
) -> torch.Tensor:
    """Return ``[len(texts), layers * 2 * hidden]`` frozen features.

    For the embedding output and every Transformer layer the CLS state and the
    masked mean over real tokens are concatenated.  Texts are truncated to the
    58-token limit, encoded once each, and batched by length so padding
    stays small on CPU.
    """

    if not texts:
        raise ValueError("no texts to encode")
    encoded = backbone.tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)
    sequences = encoded["input_ids"]
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    order = torch.argsort(lengths, stable=True).tolist()
    features: torch.Tensor | None = None
    batch_size = max(1, int(batch_size))
    backbone.model.eval()
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        width = int(lengths[rows].max())
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for row, index in enumerate(rows):
            sequence = sequences[index]
            input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, : len(sequence)] = 1
        output = backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            output_hidden_states=True,
            return_dict=True,
        )
        states = torch.stack(output.hidden_states, dim=1)  # [B, layers, T, H]
        mask = attention_mask[:, None, :, None].to(states.dtype)
        cls = states[:, :, 0]
        mean = (states * mask).sum(2) / mask.sum(2).clamp_min(1.0)
        batch = torch.cat((cls.flatten(1), mean.flatten(1)), dim=1).float()
        if features is None:
            features = torch.empty(len(texts), batch.shape[1])
        features[torch.tensor(rows)] = batch
    assert features is not None
    return features


def unique_texts(groups: Sequence[Sequence[str]]) -> tuple[list[str], list[list[int]]]:
    """Deduplicate texts; return the unique list and per-group indices into it."""

    position: dict[str, int] = {}
    texts: list[str] = []
    indices: list[list[int]] = []
    for group in groups:
        row = []
        for text in group:
            if text not in position:
                position[text] = len(texts)
                texts.append(text)
            row.append(position[text])
        indices.append(row)
    return texts, indices
