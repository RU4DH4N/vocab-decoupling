import numpy as np

COMMON = (
    "the a and to of in it was he she they we you said on for with at but not "
    "is had his her their this that from went came back home school bank road "
    "morning evening little big dog cat mum dad friend teacher asked looked "
    "played walked because then when after before again very really"
).split()
LONG = (
    "internationalisation",
    "counterrevolutionaries",
    "uncharacteristically",
    "misunderstandings",
)
OTHER = ("naïve", "café", "日本語", "über", "2026", "3.14", "don't", "o'clock")
PUNCTUATION = (",", ".", "!", "?", ";")


def documents(seed: int, count: int, words: tuple[int, int]) -> list[str]:
    rng = np.random.default_rng(seed)
    texts = []
    for _ in range(count):
        tokens = []
        for _ in range(int(rng.integers(*words))):
            roll = rng.random()
            word = (
                LONG[rng.integers(len(LONG))]
                if roll < 0.03
                else OTHER[rng.integers(len(OTHER))]
                if roll < 0.08
                else COMMON[rng.integers(len(COMMON))]
            )
            if rng.random() < 0.1:
                word += PUNCTUATION[rng.integers(len(PUNCTUATION))]
            tokens.append(word)
            if rng.random() < 0.04:
                tokens.append("\n")
        texts.append(" ".join(tokens).replace(" \n ", "\n"))
    return texts
