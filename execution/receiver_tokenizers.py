import argparse
from collections.abc import Iterator
from pathlib import Path

from data.protocol_corpus import EventCorpus
from data.tokenizers import train_bpe
from execution.layout import Layout


def tokenizer_path(output: Path, vocabulary: int) -> Path:
    return output / "extensions" / "tokenizers" / f"bpe-{vocabulary}.json"


def pieces(corpus: EventCorpus) -> Iterator[str]:
    vocab = corpus.vocab
    for unit in corpus.ids["train"]:
        yield vocab[unit]


def train(output: Path, vocabulary: int) -> None:
    corpus = EventCorpus.load(Layout(output, 0).corpus)
    tokenizer = train_bpe(vocabulary, pieces(corpus), False)
    destination = tokenizer_path(output, vocabulary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    args = parser.parse_args()
    train(args.output, args.vocabulary)


if __name__ == "__main__":
    main()
