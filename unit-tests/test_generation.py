import pytest
import torch

from data.tokenizers import train_bpe
from execution.generation import GenerationSession, SymbolBytes
from execution.sampling import GenerationLimits
from models.bpe.bpe import BPEModel
from models.protocol.lookahead import LookaheadReceiver
from models.protocol.model import ProtocolModel
from models.protocol.planner import EventPlanner
from models.protocol.receiver import Receiver
from models.protocol.sender import CoarseSender
from models.shared.symbols import START, STOP


def test_bpe_symbol_decoding_preserves_partial_utf8_and_all_byte_values():
    tokenizer = train_bpe(256, ["hello"], False)
    symbols = SymbolBytes.bpe(tokenizer)
    assert set(symbols.values) == {bytes([i]) for i in range(256)}
    for text in ("héllo 🌍", "\x00\n\t", "日本語"):
        ids = tokenizer.encode(text).ids
        assert b"".join(symbols.values[i] for i in ids) == text.encode()


@pytest.mark.parametrize(
    "change",
    [
        {"bytes": 0},
        {"symbols": True},
        {"event_bytes": -1},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_p": float("nan")},
    ],
)
def test_sampling_configuration_fails_before_generation(change):
    settings = dict(bytes=32, symbols=64, event_bytes=32, temperature=1.0, top_p=1.0)
    with pytest.raises(ValueError):
        GenerationLimits(**{**settings, **change})


def test_empty_symbol_is_not_an_ordinary_byte_emission():
    with pytest.raises(ValueError, match="nonempty bytes"):
        SymbolBytes((b"",))


def session_parts(lookahead, local=False):
    torch.manual_seed(11)
    sender = CoarseSender(
        in_dims=512,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2,
        d_meaning=8,
        codebook_seed=3,
        dropout=0.0,
        multiple_of=8,
        max_seq_len=16,
        rope_theta=10000,
        norm_eps=1e-5,
        initialiser_range=0.02,
    )
    native = BPEModel(
        vocab_size=258,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2,
        dropout=0.0,
        multiple_of=8,
        max_seq_len=128,
        rope_theta=10000,
        norm_eps=1e-5,
        initialiser_range=0.02,
        tie_word_embeddings=True,
        boundaries=[256, 257] if local else None,
    )
    model = ProtocolModel(sender, Receiver(native, 8, 4, 4, 3, 257)).eval()
    with torch.no_grad():
        for channel in model.receiver.channels:
            channel.up.weight.normal_(std=0.3)
    if not lookahead:
        return model, {}
    future = LookaheadReceiver(model.receiver, 1, 2).eval()
    with torch.no_grad():
        for layer in future.channels:
            for channel in layer:
                channel.up.weight.normal_(std=0.3)
    planner = EventPlanner(8, 8, 16, 4).eval()

    def step(previous, hidden, current):
        logits = torch.full((previous.shape[0], STOP + 1), -10.0)
        byte = 97 + current[:, 1].gt(0).long()
        stop = previous.ne(START)
        logits[torch.arange(len(byte)), torch.where(stop, STOP, byte)] = 0.0
        return logits, hidden

    planner.decoder.step = step
    return model, {
        "lookahead": future,
        "planner": planner,
        "planner_max_symbols": 4,
        "planner_width": 2,
    }


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("lookahead", [False, True])
def test_single_pass_prompt_matches_symbol_by_symbol(lookahead, local):
    model, extra = session_parts(lookahead, local)
    symbols = SymbolBytes.bytes()
    stop, start = symbols.inventory.stop, symbols.inventory.start
    prompt = [list(b"the"), list(b" cat"), list(b" sat")]
    batched = GenerationSession(model, symbols, hard=False, **extra)
    stepped = GenerationSession(model, symbols, hard=False, **extra)
    left = batched.prime(prompt)
    right = stepped.push(start)
    for event in prompt:
        for symbol in (*event, stop):
            right = stepped.push(symbol)
    torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
    assert batched.events == stepped.events
    for a, b in zip(batched.messages, stepped.messages, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    if lookahead:
        assert batched.plan is not None and stepped.plan is not None
        torch.testing.assert_close(batched.plan.messages, stepped.plan.messages)
        torch.testing.assert_close(batched.plan.log_probs, stepped.plan.log_probs)
        assert torch.equal(batched.plan.available, stepped.plan.available)
        assert batched.plan.available.any()
    for symbol in (*b" on", stop, *b" a", stop, *b" mat"):
        torch.testing.assert_close(
            batched.push(symbol), stepped.push(symbol), rtol=1e-4, atol=1e-5
        )


def test_word_local_generation_matches_whole_sequence_scoring():
    model, _ = session_parts(False, local=True)
    symbols = SymbolBytes.bytes()
    stop, start = symbols.inventory.stop, symbols.inventory.start
    stream = [start, *b"the", stop, *b" cat", stop, *b" sat", stop, *b" on"]
    session = GenerationSession(model, symbols, hard=False)
    stepped = torch.stack([session.push(symbol) for symbol in stream])
    ids = torch.tensor([stream])
    frontier = ids.eq(stop).cumsum(-1) - 1
    with torch.no_grad():
        codes, _ = model.sender.encode_trunk_vocab(session.events)
        memory = model.sender.state(codes[None])
    count = memory.shape[1]
    events = torch.arange(count).expand(1, ids.shape[1], count)
    candidates = memory[:, None].expand(1, ids.shape[1], count, memory.shape[-1])
    with torch.no_grad():
        whole = model.receiver(ids, candidates, events, frontier, hard=False).logits[0]
    torch.testing.assert_close(stepped, whole, rtol=1e-4, atol=1e-5)


def test_word_local_receiver_cannot_see_earlier_words():
    stop, start = 256, 257
    for local in (True, False):
        model, _ = session_parts(False, local)
        native = model.receiver.native
        first = torch.tensor([[start, *b"the", stop, *b" cat"]])
        second = torch.tensor([[start, *b"dog", stop, *b" cat"]])
        with torch.no_grad():
            changed = not torch.allclose(native(first)[0, 4:], native(second)[0, 4:])
        assert changed is not local
