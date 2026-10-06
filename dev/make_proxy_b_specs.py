"""Sample independent attribute specs for the LLM-written proxy corpus (proxy B).

Each spec describes one Russian message that a generator agent must write.
Attributes are sampled independently so that any single attribute can later be
used as a hidden episode rule without being confounded by the others.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "proxy_b"

TOPICS = [
    "еда и кулинария",
    "путешествия и транспорт",
    "работа и офис",
    "здоровье и медицина",
    "техника и гаджеты",
    "деньги и банки",
    "дом и ремонт",
    "спорт и фитнес",
    "учёба и школа",
    "домашние животные",
    "погода и природа",
    "покупки и доставка",
    "кино, музыка и книги",
    "семья и друзья",
]
SENTIMENT = ["позитивная", "негативная", "нейтральная"]
FORM = ["вопрос", "утверждение", "просьба"]
TENSE = ["прошедшее", "настоящее", "будущее"]
PERSON = ["1 лицо (я/мы)", "2 лицо (ты/вы)", "3 лицо (он/она/они/кто-то)"]

STYLES = [
    "сообщения в мессенджере другу",
    "обращения в службу поддержки",
    "посты на форуме",
    "заметки в личном дневнике",
    "реплики из разговора с коллегой",
    "комментарии под постом в соцсети",
    "SMS родственникам",
    "отзывы и жалобы",
    "голосовые сообщения, переведённые в текст",
    "записки соседям и объявления",
    "сообщения в рабочем чате",
    "реплики в семейном чате",
]

CHUNKS = 12
PER_CHUNK = 250


def main() -> None:
    rng = random.Random(20261003)
    OUT.mkdir(parents=True, exist_ok=True)
    for chunk in range(CHUNKS):
        rows = []
        for index in range(PER_CHUNK):
            form = rng.choice(FORM)
            spec = {
                "id": f"b{chunk:02d}_{index:03d}",
                "topic": rng.choice(TOPICS),
                "sentiment": rng.choice(SENTIMENT),
                "form": form,
                "tense": None if form == "просьба" else rng.choice(TENSE),
                "person": rng.choice(PERSON),
                "has_number": rng.random() < 0.5,
                "has_negation": rng.random() < 0.5,
                "mentions_person": rng.random() < 0.5,
                "urgent": rng.random() < 0.5,
                "colloquial": rng.random() < 0.5,
                "mentions_place": rng.random() < 0.5,
                "long": rng.random() < 0.5,
            }
            rows.append(spec)
        with (OUT / f"spec_{chunk:02d}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "styles.json").open("w", encoding="utf-8") as stream:
        json.dump(STYLES, stream, ensure_ascii=False, indent=1)
    print(f"wrote {CHUNKS} spec chunks x {PER_CHUNK}")


if __name__ == "__main__":
    main()
