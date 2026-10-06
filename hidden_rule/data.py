"""Episode JSONL reading and prediction writing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

SUPPORT_SIZE = 8
QUERY_COUNT = 4


class EpisodeFormatError(ValueError):
    """Raised when an episode file does not match the expected schema."""


@dataclass(frozen=True)
class Item:
    item_id: str
    text: str
    label: int | None


@dataclass(frozen=True)
class Episode:
    episode_id: str
    support: tuple[Item, ...]
    queries: tuple[Item, ...]


def _parse_item(raw: Any, id_field: str, require_label: bool, where: str) -> Item:
    if not isinstance(raw, dict):
        raise EpisodeFormatError(f"{where}: expected an object")
    item_id, text, label = raw.get(id_field), raw.get("text"), raw.get("label")
    if not isinstance(item_id, str) or not item_id:
        raise EpisodeFormatError(f"{where}: {id_field} must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise EpisodeFormatError(f"{where}: text must be a non-empty string")
    if label is None and not require_label:
        return Item(item_id, text, None)
    if isinstance(label, bool) or label not in (0, 1):
        raise EpisodeFormatError(f"{where}: label must be 0 or 1")
    return Item(item_id, text, label)


def parse_episode(raw: Any, *, require_query_labels: bool = False, where: str = "episode") -> Episode:
    if not isinstance(raw, dict):
        raise EpisodeFormatError(f"{where}: expected an object")
    episode_id = raw.get("episode_id")
    support, queries = raw.get("support"), raw.get("queries")
    if not isinstance(episode_id, str) or not episode_id:
        raise EpisodeFormatError(f"{where}: episode_id must be a non-empty string")
    if not isinstance(support, list) or len(support) != SUPPORT_SIZE:
        raise EpisodeFormatError(f"{where}: support must contain {SUPPORT_SIZE} items")
    if not isinstance(queries, list) or len(queries) != QUERY_COUNT:
        raise EpisodeFormatError(f"{where}: queries must contain {QUERY_COUNT} items")
    support_items = tuple(
        _parse_item(item, "example_id", True, f"{where}.support[{i}]") for i, item in enumerate(support)
    )
    if sum(item.label for item in support_items) != SUPPORT_SIZE // 2:
        raise EpisodeFormatError(f"{where}: support labels must be balanced 4/4")
    query_items = tuple(
        _parse_item(item, "query_id", require_query_labels, f"{where}.queries[{i}]")
        for i, item in enumerate(queries)
    )
    if len({item.label is None for item in query_items}) != 1:
        raise EpisodeFormatError(f"{where}: query labels must be given for all queries or none")
    ids = [item.item_id for item in support_items + query_items]
    if len(ids) != len(set(ids)):
        raise EpisodeFormatError(f"{where}: item ids must be unique within an episode")
    return Episode(episode_id, support_items, query_items)


def read_episodes(path: str | Path, *, require_query_labels: bool = False) -> list[Episode]:
    episodes: list[Episode] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise EpisodeFormatError(f"{where}: invalid JSON") from error
            episode = parse_episode(raw, require_query_labels=require_query_labels, where=where)
            if episode.episode_id in seen:
                raise EpisodeFormatError(f"{where}: duplicate episode_id")
            seen.add(episode.episode_id)
            episodes.append(episode)
    if not episodes:
        raise EpisodeFormatError(f"{path}: no episodes")
    return episodes


def write_predictions(episodes: Sequence[Episode], probabilities: torch.Tensor, path: str | Path) -> None:
    """Write one row per query; ``label`` is 1 exactly when ``probability >= 0.5``."""

    if tuple(probabilities.shape) != (len(episodes), QUERY_COUNT):
        raise ValueError("probabilities must have shape [episodes, 4]")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for episode, row in zip(episodes, probabilities.tolist()):
            for query, probability in zip(episode.queries, row):
                record = {
                    "episode_id": episode.episode_id,
                    "query_id": query.item_id,
                    "label": int(probability >= 0.5),
                    "probability": float(probability),
                }
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
