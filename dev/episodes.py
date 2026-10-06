"""Build episodic JSONL splits from the proxy corpora.

Every split is written twice: as episode JSONL (what ``hidden_rule`` reads)
and as a compact index file used by the research harness.  Phrase pools of the
train and test splits are disjoint; ``test_unseen`` additionally uses only
rules whose attribute family never appears as a rule in train.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

DEV = Path(__file__).resolve().parent
ROOT = DEV.parent

Rule = tuple[str, str, Callable[[dict], int | None]]


def _eq(key: str, value) -> Callable[[dict], int | None]:
    return lambda a: None if a.get(key) is None else int(a[key] == value)


def _in(key: str, values) -> Callable[[dict], int | None]:
    return lambda a: None if a.get(key) is None else int(a[key] in values)


def _contrast(key: str, positive, negative) -> Callable[[dict], int | None]:
    def rule(a: dict) -> int | None:
        if a.get(key) == positive:
            return 1
        if a.get(key) == negative:
            return 0
        return None

    return rule


def _bool(key: str) -> Callable[[dict], int | None]:
    return lambda a: None if a.get(key) is None else int(bool(a[key]))


def rules_a() -> list[Rule]:
    rules: list[Rule] = []
    for category in ["animal", "person", "vehicle", "plant", "food", "object"]:
        rules.append((f"category={category}", "category", _eq("category", category)))
    rules.append(("category∈animate", "category", _in("category", {"animal", "person"})))
    rules.append(("category∈edible_or_plant", "category", _in("category", {"food", "plant"})))
    for color in ["red", "blue", "green", "yellow", "black", "white"]:
        rules.append((f"color={color}", "color", _eq("color", color)))
    rules.append(("color warm/cold", "color", lambda a: 1 if a["color"] in {"red", "yellow"} else (0 if a["color"] in {"blue", "green"} else None)))
    rules.append(("size=big", "size", _eq("size", "big")))
    rules.append(("plural", "plural", _bool("plural")))
    for tense in ["past", "present", "future"]:
        rules.append((f"tense={tense}", "tense", _eq("tense", tense)))
    rules.append(("tense past/future", "tense", _contrast("tense", "past", "future")))
    rules.append(("negation", "negation", _bool("negation")))
    rules.append(("question", "question", _bool("question")))
    rules.append(("daytime=dark", "daytime", _eq("daytime", "dark")))
    rules.append(("season=winter", "season", _eq("season", "winter")))
    rules.append(("season warm", "season", _in("season", {"spring", "summer"})))
    rules.append(("location=indoor", "location", _eq("location", "indoor")))
    rules.append(("mood good/bad", "mood", _contrast("mood", "good", "bad")))
    rules.append(("mood present", "mood", lambda a: int(a["mood"] != "none")))
    rules.append(("verb=dynamic", "verb_class", _eq("verb_class", "dynamic")))
    return rules


HELDOUT_A = {"color", "season", "mood", "negation"}

TOPICS_B = [
    "еда и кулинария", "путешествия и транспорт", "работа и офис", "здоровье и медицина",
    "техника и гаджеты", "деньги и банки", "дом и ремонт", "спорт и фитнес",
    "учёба и школа", "домашние животные", "погода и природа", "покупки и доставка",
    "кино, музыка и книги", "семья и друзья",
]
HELDOUT_TOPICS_B = {"здоровье и медицина", "домашние животные", "кино, музыка и книги", "деньги и банки"}


def rules_b() -> list[Rule]:
    rules: list[Rule] = []
    for topic in TOPICS_B:
        family = "topic_heldout" if topic in HELDOUT_TOPICS_B else "topic"
        rules.append((f"topic={topic}", family, _eq("topic", topic)))
    rules.append(("sentiment=негативная", "sentiment", _eq("sentiment", "негативная")))
    rules.append(("sentiment=позитивная", "sentiment", _eq("sentiment", "позитивная")))
    rules.append(("sentiment pos/neg", "sentiment", _contrast("sentiment", "позитивная", "негативная")))
    for form in ["вопрос", "утверждение", "просьба"]:
        rules.append((f"form={form}", "form", _eq("form", form)))
    for tense in ["прошедшее", "будущее"]:
        rules.append((f"tense={tense}", "tense", _eq("tense", tense)))
    rules.append(("tense past/future", "tense", _contrast("tense", "прошедшее", "будущее")))
    for person in ["1 лицо (я/мы)", "2 лицо (ты/вы)", "3 лицо (он/она/они/кто-то)"]:
        rules.append((f"person={person}", "person", _eq("person", person)))
    for key in ["has_number", "has_negation", "mentions_person", "urgent", "colloquial", "mentions_place", "long"]:
        rules.append((key, key, _bool(key)))
    return rules


HELDOUT_B = {"topic_heldout", "person", "urgent", "mentions_place"}


def load_corpus(name: str) -> list[dict]:
    if name == "a":
        path = DEV / "proxy_a" / "corpus.jsonl"
    elif name == "b":
        path = DEV / "proxy_b" / "corpus.jsonl"
    else:
        raise ValueError(name)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _hex(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.getrandbits(48):012x}"


def sample_episodes(
    corpus: list[dict],
    pool: list[int],
    rules: list[Rule],
    count: int,
    rng: random.Random,
    query_mode: str,
    flip_rate: float,
) -> list[dict]:
    by_rule = []
    for name, family, fn in rules:
        pos = [i for i in pool if fn(corpus[i]["attrs"]) == 1]
        neg = [i for i in pool if fn(corpus[i]["attrs"]) == 0]
        if len(pos) >= 12 and len(neg) >= 12:
            by_rule.append((name, family, pos, neg))
    episodes = []
    for _ in range(count):
        name, family, pos, neg = rng.choice(by_rule)
        flipped = rng.random() < flip_rate
        if flipped:
            pos, neg = neg, pos
        if query_mode == "balanced":
            q_labels = [1, 1, 0, 0]
        elif query_mode == "bern":
            q_labels = [int(rng.random() < 0.5) for _ in range(4)]
        else:
            raise ValueError(query_mode)
        n_pos_q = sum(q_labels)
        pos_items = rng.sample(pos, 4 + n_pos_q)
        neg_items = rng.sample(neg, 4 + 4 - n_pos_q)
        support = [(i, 1) for i in pos_items[:4]] + [(i, 0) for i in neg_items[:4]]
        queries = [(i, 1) for i in pos_items[4:]] + [(i, 0) for i in neg_items[4:]]
        rng.shuffle(support)
        rng.shuffle(queries)
        episodes.append(
            {
                "episode_id": _hex(rng, "ep"),
                "rule": name + (" [flipped]" if flipped else ""),
                "family": family,
                "support": [{"example_id": _hex(rng, "sx"), "idx": i, "label": y} for i, y in support],
                "queries": [{"query_id": _hex(rng, "qx"), "idx": i, "label": y} for i, y in queries],
            }
        )
    return episodes


def write_split(corpus: list[dict], episodes: list[dict], out_dir: Path, name: str, with_labels: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
        for ep in episodes:
            row = {
                "episode_id": ep["episode_id"],
                "support": [
                    {"example_id": s["example_id"], "text": corpus[s["idx"]]["text"], "label": s["label"]}
                    for s in ep["support"]
                ],
                "queries": [
                    {"query_id": q["query_id"], "text": corpus[q["idx"]]["text"], **({"label": q["label"]} if with_labels else {})}
                    for q in ep["queries"]
                ],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not with_labels:
        with (out_dir / f"{name}.labels.jsonl").open("w", encoding="utf-8") as stream:
            for ep in episodes:
                for q in ep["queries"]:
                    stream.write(json.dumps({"episode_id": ep["episode_id"], "query_id": q["query_id"], "label": q["label"], "rule": ep["rule"], "family": ep["family"]}, ensure_ascii=False) + "\n")
    with (out_dir / f"{name}.index.json").open("w", encoding="utf-8") as stream:
        json.dump(episodes, stream, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["a", "b"], required=True)
    parser.add_argument("--query-mode", choices=["balanced", "bern"], default="bern")
    parser.add_argument("--flip-rate", type=float, default=0.0)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    rules = rules_a() if args.corpus == "a" else rules_b()
    heldout = HELDOUT_A if args.corpus == "a" else HELDOUT_B
    rng = random.Random(args.seed)
    order = list(range(len(corpus)))
    rng.shuffle(order)
    cut = int(0.7 * len(order))
    train_pool, test_pool = sorted(order[:cut]), sorted(order[cut:])
    seen_rules = [r for r in rules if r[1] not in heldout]
    unseen_rules = [r for r in rules if r[1] in heldout]

    tag = args.tag or f"{args.corpus}_{args.query_mode}" + (f"_flip{args.flip_rate:g}" if args.flip_rate else "")
    out_dir = ROOT / "runs" / "proxy" / tag
    train = sample_episodes(corpus, train_pool, seen_rules, args.train_episodes, rng, args.query_mode, args.flip_rate)
    valid = sample_episodes(corpus, train_pool, seen_rules, 600, rng, args.query_mode, args.flip_rate)
    test_seen = sample_episodes(corpus, test_pool, seen_rules, args.test_episodes, rng, args.query_mode, args.flip_rate)
    test_unseen = sample_episodes(corpus, test_pool, unseen_rules, args.test_episodes, rng, args.query_mode, args.flip_rate)
    write_split(corpus, train, out_dir, "train", True)
    write_split(corpus, valid, out_dir, "valid", True)
    write_split(corpus, test_seen, out_dir, "test_seen", False)
    write_split(corpus, test_unseen, out_dir, "test_unseen", False)
    print(json.dumps({
        "out": str(out_dir),
        "seen_rules": len(seen_rules),
        "unseen_rules": len(unseen_rules),
        "train_pool": len(train_pool),
        "test_pool": len(test_pool),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
