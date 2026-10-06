"""Episodic few-shot head: learned projection plus a support-gated LDA scorer."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

SUPPORT_SIZE = 8
QUERY_COUNT = 4


class ParticipantModule(nn.Module):
    """Score queries against the labelled support set of their own episode.

    Frozen sentence features are standardized and projected to a small space.
    For each episode the class prototypes and pooled within-class variance of
    the support define a diagonal LDA direction; a gate driven by each
    dimension's support t-statistic keeps only the dimensions that separate the
    two support classes, which is how the head "switches" to the rule active in
    that episode.  The positive-query-count prior learned from train is kept as
    a buffer for joint decoding of the four queries.
    """

    def __init__(
        self,
        hidden_size: int = 312,
        *,
        input_dim: int | None = None,
        projection_dim: int = 48,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        input_dim = int(hidden_size if input_dim is None else input_dim)
        self.input_dim = input_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.register_buffer("count_log_prior", torch.full((QUERY_COUNT + 1,), -math.log(QUERY_COUNT + 1)))
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(input_dim, projection_dim)
        self.log_prior_var = nn.Parameter(torch.zeros(projection_dim))
        self.shrink = nn.Parameter(torch.tensor(0.0))
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> ParticipantModule:
        """Rebuild the module with dimensions inferred from a saved adapter."""

        weight = state["projection.weight"]
        module = cls(input_dim=int(weight.shape[1]), projection_dim=int(weight.shape[0]))
        module.load_state_dict(dict(state), strict=True)
        return module

    def set_feature_stats(self, features: torch.Tensor) -> None:
        self.feature_mean.copy_(features.mean(0))
        self.feature_std.copy_(features.std(0).nan_to_num(1.0) + 1e-4)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features - self.feature_mean) / self.feature_std
        return self.projection(self.dropout(standardized))

    def forward(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits ``[batch, queries]``.

        ``query_embeddings`` is ``[batch, queries, input_dim]``;
        ``support_embeddings`` is ``[batch, support, input_dim]`` with binary
        ``support_labels`` ``[batch, support]``.  Without a support set no rule
        is observable, so the logits are uninformative zeros.
        """

        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape [batch, queries, hidden]")
        if support_embeddings is None or support_labels is None:
            return query_embeddings.new_zeros(query_embeddings.shape[:2])
        queries = self.embed(query_embeddings)
        support = self.embed(support_embeddings)
        labels = support_labels.to(support.dtype).unsqueeze(-1)
        positives = labels.sum(1).clamp_min(1.0)
        negatives = (1.0 - labels).sum(1).clamp_min(1.0)
        center_pos = (support * labels).sum(1) / positives
        center_neg = (support * (1.0 - labels)).sum(1) / negatives
        residual = support - (labels * center_pos[:, None] + (1.0 - labels) * center_neg[:, None])
        dof = (positives + negatives - 2.0).clamp_min(1.0)
        variance = (residual**2).sum(1) / dof
        weight = torch.sigmoid(self.shrink)
        variance = (1.0 - weight) * variance + weight * self.log_prior_var.exp()[None]
        variance = variance + 1e-4
        difference = center_pos - center_neg
        t_statistic = difference.abs() / variance.sqrt()
        gate = torch.sigmoid(self.gate_scale * t_statistic + self.gate_bias)
        direction = gate * difference / variance
        midpoint = 0.5 * (center_pos + center_neg)
        logits = torch.einsum("bqd,bd->bq", queries - midpoint[:, None], direction)
        return self.scale * logits / math.sqrt(queries.shape[-1]) + self.bias


def _label_configurations(count: int) -> torch.Tensor:
    return torch.tensor(
        [[(mask >> bit) & 1 for bit in range(count)] for mask in range(2**count)],
        dtype=torch.float32,
    )


def joint_decode(logits: torch.Tensor, count_log_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact-match-optimal decoding of each episode's queries.

    Per-query posteriors already include an iid Bernoulli(1/2) prior, so the
    learned prior over the number of positive queries enters relative to the
    binomial.  Returns MAP labels ``[batch, queries]`` and probabilities that
    are the joint-model marginals, nudged so ``label == (probability >= 0.5)``.
    """

    logits = logits.nan_to_num(0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    count = logits.shape[-1]
    configs = _label_configurations(count).to(logits)
    binomial = torch.tensor([math.comb(count, k) for k in range(count + 1)], dtype=logits.dtype)
    prior = count_log_prior.to(logits)
    if prior.numel() != count + 1:
        prior = torch.zeros(count + 1, dtype=logits.dtype)
    adjust = (prior - binomial.log())[configs.sum(1).long()]
    scores = F.logsigmoid(logits) @ configs.T + F.logsigmoid(-logits) @ (1 - configs).T + adjust
    labels = configs[scores.argmax(1)].long()
    posterior = torch.softmax(scores, dim=1)
    marginals = (posterior @ configs).nan_to_num(0.5).clamp(0.0, 1.0)
    below = torch.nextafter(torch.tensor(0.5, dtype=marginals.dtype), torch.tensor(0.0, dtype=marginals.dtype))
    marginals = torch.where(labels.bool(), marginals.clamp_min(0.5), marginals.clamp_max(below))
    return labels, marginals
