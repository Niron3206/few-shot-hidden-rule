"""Few-shot classification of Russian texts by a hidden per-episode rule."""

from hidden_rule.model import EpisodeClassifier, joint_decode

__all__ = ["EpisodeClassifier", "joint_decode"]
