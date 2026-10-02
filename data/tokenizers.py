from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_bpe(vocab_size: int, texts: Iterable[str], use_regex: bool) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=use_regex
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
