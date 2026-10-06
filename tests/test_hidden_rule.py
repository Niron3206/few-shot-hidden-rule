"""Fast checks that need neither the model weights nor training data."""

from __future__ import annotations

import json

import pytest
import torch

from hidden_rule.data import EpisodeFormatError, read_episodes, write_predictions
from hidden_rule.model import EpisodeClassifier, joint_decode


def _episode(with_query_labels: bool = True) -> dict:
    return {
        "episode_id": "ep1",
        "support": [{"example_id": f"s{i}", "text": f"пример {i}", "label": i % 2} for i in range(8)],
        "queries": [
            {"query_id": f"q{i}", "text": f"запрос {i}", **({"label": i % 2} if with_query_labels else {})}
            for i in range(4)
        ],
    }


def _write(tmp_path, episode: dict):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("with_labels", [True, False])
def test_reader_accepts_labelled_and_unlabelled_queries(tmp_path, with_labels):
    episodes = read_episodes(_write(tmp_path, _episode(with_labels)), require_query_labels=with_labels)
    assert len(episodes[0].support) == 8 and len(episodes[0].queries) == 4
    assert (episodes[0].queries[0].label is None) is not with_labels


def test_reader_rejects_unbalanced_support(tmp_path):
    episode = _episode()
    for item in episode["support"]:
        item["label"] = 1
    with pytest.raises(EpisodeFormatError, match="balanced"):
        read_episodes(_write(tmp_path, episode))


def test_classifier_is_symmetric_to_label_swap():
    torch.manual_seed(0)
    model = EpisodeClassifier(16, projection_dim=8).eval()
    support, queries = torch.randn(3, 8, 16), torch.randn(3, 4, 16)
    labels = torch.tensor([[1, 0] * 4] * 3)
    logits = model(queries, support, labels)
    assert logits.shape == (3, 4)
    assert torch.allclose(model(queries, support, 1 - labels), -logits, atol=1e-5)


def test_adapter_round_trip():
    model = EpisodeClassifier(16, projection_dim=8)
    restored = EpisodeClassifier.from_state_dict(model.state_dict())
    assert restored.projection.weight.shape == (8, 16)


def test_joint_decode_respects_count_prior_and_threshold():
    logits = torch.tensor([[3.0, 2.5, 2.0, -1.0]])
    only_two = torch.log(torch.tensor([1e-9, 1e-9, 1.0, 1e-9, 1e-9]))
    labels, probabilities = joint_decode(logits, only_two)
    assert labels.tolist() == [[1, 1, 0, 0]]
    assert torch.equal(labels.bool(), probabilities >= 0.5)


def test_written_labels_follow_probabilities(tmp_path):
    episodes = read_episodes(_write(tmp_path, _episode(False)))
    output = tmp_path / "predictions.jsonl"
    write_predictions(episodes, torch.tensor([[0.5, 0.49, 1.0, 0.0]], dtype=torch.float64), output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["label"] for row in rows] == [1, 0, 1, 0]
    assert {row["query_id"] for row in rows} == {"q0", "q1", "q2", "q3"}
