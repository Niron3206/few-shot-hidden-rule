"""Template-generated proxy corpus A with exact, independent attributes.

Sentences describe scenes ("К сожалению, зимой утром большие красные лодки не
стояли в подвале.").  Every attribute is sampled independently, so each one can
serve as a hidden episode rule.  This corpus complements the LLM-written proxy B
and is used only for local research, not by the solution itself.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "proxy_a" / "corpus.jsonl"

# noun: (singular, plural, gender)
NOUNS = {
    "animal": [
        ("кот", "коты", "m"), ("собака", "собаки", "f"), ("лошадь", "лошади", "f"),
        ("волк", "волки", "m"), ("лиса", "лисы", "f"), ("медведь", "медведи", "m"),
        ("заяц", "зайцы", "m"), ("корова", "коровы", "f"), ("белка", "белки", "f"),
        ("олень", "олени", "m"), ("сова", "совы", "f"), ("ёж", "ежи", "m"),
    ],
    "person": [
        ("врач", "врачи", "m"), ("учительница", "учительницы", "f"),
        ("повар", "повара", "m"), ("студентка", "студентки", "f"),
        ("водитель", "водители", "m"), ("бабушка", "бабушки", "f"),
        ("почтальон", "почтальоны", "m"), ("художница", "художницы", "f"),
        ("инженер", "инженеры", "m"), ("продавщица", "продавщицы", "f"),
        ("рыбак", "рыбаки", "m"), ("медсестра", "медсёстры", "f"),
    ],
    "vehicle": [
        ("автобус", "автобусы", "m"), ("машина", "машины", "f"),
        ("поезд", "поезда", "m"), ("велосипед", "велосипеды", "m"),
        ("лодка", "лодки", "f"), ("трамвай", "трамваи", "m"),
        ("самолёт", "самолёты", "m"), ("грузовик", "грузовики", "m"),
        ("электричка", "электрички", "f"), ("мотоцикл", "мотоциклы", "m"),
        ("яхта", "яхты", "f"), ("тележка", "тележки", "f"),
    ],
    "plant": [
        ("дерево", "деревья", "n"), ("роза", "розы", "f"), ("берёза", "берёзы", "f"),
        ("кактус", "кактусы", "m"), ("дуб", "дубы", "m"), ("подсолнух", "подсолнухи", "m"),
        ("куст", "кусты", "m"), ("ромашка", "ромашки", "f"), ("ель", "ели", "f"),
        ("клён", "клёны", "m"), ("тюльпан", "тюльпаны", "m"), ("папоротник", "папоротники", "m"),
    ],
    "food": [
        ("пирог", "пироги", "m"), ("суп", "супы", "m"), ("яблоко", "яблоки", "n"),
        ("торт", "торты", "m"), ("хлеб", "хлеба", "m"), ("каша", "каши", "f"),
        ("пицца", "пиццы", "f"), ("арбуз", "арбузы", "m"), ("сыр", "сыры", "m"),
        ("булочка", "булочки", "f"), ("пельмень", "пельмени", "m"), ("груша", "груши", "f"),
    ],
    "object": [
        ("молоток", "молотки", "m"), ("ключ", "ключи", "m"), ("лампа", "лампы", "f"),
        ("телефон", "телефоны", "m"), ("книга", "книги", "f"), ("зонт", "зонты", "m"),
        ("чашка", "чашки", "f"), ("стул", "стулья", "m"), ("ноутбук", "ноутбуки", "m"),
        ("ваза", "вазы", "f"), ("кресло", "кресла", "n"), ("коробка", "коробки", "f"),
    ],
}

# adjective forms: m, f, n, pl
COLORS = {
    "red": ("красный", "красная", "красное", "красные"),
    "blue": ("синий", "синяя", "синее", "синие"),
    "green": ("зелёный", "зелёная", "зелёное", "зелёные"),
    "yellow": ("жёлтый", "жёлтая", "жёлтое", "жёлтые"),
    "black": ("чёрный", "чёрная", "чёрное", "чёрные"),
    "white": ("белый", "белая", "белое", "белые"),
}
SIZES = {
    "big": [
        ("большой", "большая", "большое", "большие"),
        ("огромный", "огромная", "огромное", "огромные"),
        ("крупный", "крупная", "крупное", "крупные"),
    ],
    "small": [
        ("маленький", "маленькая", "маленькое", "маленькие"),
        ("крошечный", "крошечная", "крошечное", "крошечные"),
        ("небольшой", "небольшая", "небольшое", "небольшие"),
    ],
}
# verb: class, present(sg, pl), past(m, f, n, pl), future(sg, pl)
VERBS = [
    ("static", ("стоит", "стоят"), ("стоял", "стояла", "стояло", "стояли"), ("будет стоять", "будут стоять")),
    ("static", ("лежит", "лежат"), ("лежал", "лежала", "лежало", "лежали"), ("будет лежать", "будут лежать")),
    ("static", ("находится", "находятся"), ("находился", "находилась", "находилось", "находились"), ("будет находиться", "будут находиться")),
    ("static", ("остаётся", "остаются"), ("оставался", "оставалась", "оставалось", "оставались"), ("останется", "останутся")),
    ("dynamic", ("движется", "движутся"), ("двигался", "двигалась", "двигалось", "двигались"), ("будет двигаться", "будут двигаться")),
    ("dynamic", ("появляется", "появляются"), ("появился", "появилась", "появилось", "появились"), ("появится", "появятся")),
    ("dynamic", ("исчезает", "исчезают"), ("исчез", "исчезла", "исчезло", "исчезли"), ("исчезнет", "исчезнут")),
    ("dynamic", ("падает", "падают"), ("упал", "упала", "упало", "упали"), ("упадёт", "упадут")),
]
LOCATIONS = {
    "indoor": [
        "в комнате", "на кухне", "в подвале", "в магазине", "в гараже", "в коридоре",
        "в музее", "в спортзале", "на складе", "в библиотеке",
    ],
    "outdoor": [
        "в лесу", "на улице", "в поле", "у реки", "на площади", "в парке",
        "на берегу моря", "в горах", "на крыше", "во дворе",
    ],
}
DAYTIME = {"light": ["утром", "днём", "на рассвете"], "dark": ["вечером", "ночью", "в полночь"]}
SEASON = {"winter": "зимой", "spring": "весной", "summer": "летом", "autumn": "осенью"}
MOOD = {"good": ["К счастью,", "Как здорово, что", "Радостно, что"], "bad": ["К сожалению,", "Жаль, что", "Обидно, что"], "none": [""]}


def build(rng: random.Random, attrs: dict) -> str:
    noun_sg, noun_pl, gender = rng.choice(NOUNS[attrs["category"]])
    plural = attrs["plural"]
    form_index = 3 if plural else {"m": 0, "f": 1, "n": 2}[gender]
    color = COLORS[attrs["color"]][form_index]
    size = rng.choice(SIZES[attrs["size"]])[form_index]
    verb = rng.choice([verb for verb in VERBS if verb[0] == attrs["verb_class"]])
    if attrs["tense"] == "past":
        verb_form = verb[2][form_index]
    elif attrs["tense"] == "present":
        verb_form = verb[1][1 if plural else 0]
    else:
        verb_form = verb[3][1 if plural else 0]
    if attrs["negation"]:
        verb_form = "не " + verb_form
    noun = noun_pl if plural else noun_sg
    location = rng.choice(LOCATIONS[attrs["location"]])
    daytime = rng.choice(DAYTIME[attrs["daytime"]])
    season = SEASON[attrs["season"]]
    adjectives = [size, color]
    rng.shuffle(adjectives)
    circumstances = [daytime, season]
    rng.shuffle(circumstances)
    body = f"{circumstances[0]} {circumstances[1]} {adjectives[0]} {adjectives[1]} {noun} {verb_form} {location}"
    if rng.random() < 0.5:
        body = f"{adjectives[0]} {adjectives[1]} {noun} {circumstances[0]} {verb_form} {location} {circumstances[1]}"
    mood = rng.choice(MOOD[attrs["mood"]])
    if attrs["question"]:
        lead = "Правда ли, что" if not mood else mood
        if mood.endswith(","):
            text = f"{mood} {body}?"
        else:
            text = f"{lead} {body}?"
    else:
        text = f"{mood} {body}." if mood else f"{body}."
    text = text.strip()
    return text[0].upper() + text[1:]


def main() -> None:
    rng = random.Random(4242)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows = []
    while len(rows) < 6000:
        attrs = {
            "category": rng.choice(list(NOUNS)),
            "color": rng.choice(list(COLORS)),
            "size": rng.choice(list(SIZES)),
            "plural": rng.random() < 0.5,
            "tense": rng.choice(["past", "present", "future"]),
            "negation": rng.random() < 0.5,
            "question": rng.random() < 0.5,
            "daytime": rng.choice(list(DAYTIME)),
            "season": rng.choice(list(SEASON)),
            "location": rng.choice(list(LOCATIONS)),
            "mood": rng.choice(list(MOOD)),
            "verb_class": rng.choice(["static", "dynamic"]),
        }
        text = build(rng, attrs)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"id": f"a{len(rows):05d}", "text": text, "attrs": attrs})
    with OUT.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} sentences to {OUT}")
    for row in rows[:8]:
        print(row["text"])


if __name__ == "__main__":
    main()
