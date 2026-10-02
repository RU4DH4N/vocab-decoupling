from data.tokenizers import load_tokenizer, train_bpe

TEXT = (
    "The quick brown fox jumps over the lazy dog.\n\nIt was the best of times,\n"
    "it was the worst of times.  Ünïcödé — and 日本語 too!\n"
    "[ Beeping ]\n####[Jazzy Solo ] ##hash ▁marker\n" * 40
)


def test_bpe_is_lossless_and_survives_saving(tmp_path):
    tokenizer = train_bpe(400, [TEXT], True)
    ids = tokenizer.encode(TEXT).ids
    assert tokenizer.decode(ids) == TEXT
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    loaded = load_tokenizer(tmp_path / "tokenizer.json")
    assert loaded.encode(TEXT).ids == ids
    assert loaded.decode(ids) == TEXT


def test_bpe_starts_from_every_byte():
    tokenizer = train_bpe(256, ["a"], False)
    assert tokenizer.get_vocab_size() == 256
    assert tokenizer.decode(tokenizer.encode("日本").ids) == "日本"
