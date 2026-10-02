from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.communication import ReceiverUnits
from data.corpus import to_units
from data.protocol_corpus import EventCorpus
from data.tokenizers import train_bpe
from execution.generation import (
    GenerationSession,
    SessionLimitError,
    SymbolBytes,
    generate_protocol,
)
from execution.protocol_evaluation import evaluate_controls
from execution.protocol_stages import StageConfig, StageTrainer
from execution.sampling import GenerationLimits
from models.bpe.bpe import BPEModel
from models.protocol.model import ProtocolModel
from models.protocol.receiver import Receiver
from models.protocol.sender import CoarseSender


def native_model(inventory):
    return BPEModel(
        vocab_size=inventory.inputs,
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
    )


@pytest.mark.parametrize("kind", ["bytes", "bpe"])
def test_native_pretraining_then_all_interface_stages_on_text(
    kind, tmp_path, monkeypatch
):
    torch.manual_seed(41)
    train = [
        "Dave checked his bank balance before buying food.",
        "Sam walked to the bank after lunch.",
    ]
    selection = ["Alice checked her bank balance before work."]
    corpus = EventCorpus(train, selection, max_bytes=31)
    if kind == "bytes":
        table = ReceiverUnits.bytes(corpus.vocab)
        symbols = SymbolBytes.bytes()
    else:
        tokenizer = train_bpe(
            280,
            [u for doc in train for u in to_units(doc, max_bytes=31)],
            False,
        )
        table = ReceiverUnits.bpe(corpus.vocab, tokenizer)
        symbols = SymbolBytes.bpe(tokenizer)
    for unit, pieces in zip(corpus.vocab, table.pieces, strict=True):
        assert b"".join(symbols.values[i] for i in pieces) == unit.encode("utf-8")
    rng = np.random.default_rng(23)

    def next_batch(split):
        rows = corpus.windows(split, events=3, batch=2, rng=rng)
        return table.batch(
            rows, device="cpu", max_receiver_steps=128, candidate_window=3
        )

    native = native_model(table.inventory)
    optimizer = torch.optim.AdamW(native.parameters(), lr=0.002)
    for _ in range(2):
        batch = next_batch("train")
        optimizer.zero_grad(set_to_none=True)
        logits = native(batch.receiver_ids)[..., : table.inventory.outputs]
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        loss.backward()
        optimizer.step()
    native_state = {name: p.detach().clone() for name, p in native.named_parameters()}
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
        max_seq_len=8,
        rope_theta=10000,
        norm_eps=1e-5,
        initialiser_range=0.02,
    )
    sender.build_table(corpus.vocab)
    receiver = Receiver(native, 8, 4, 4, 3, output_symbols=table.inventory.outputs)
    model = ProtocolModel(sender, receiver)

    for stage in ("alignment", "communication", "joint", "trunk"):
        config = StageConfig(
            name=stage,
            steps=2,
            learning_rate=0.002,
            weight_decay=0.01,
            alignment_weight=1.0 if stage == "alignment" else 0.0,
            grad_clip=1.0,
            warmup_steps=0,
            precision="fp32",
        )
        trainer = StageTrainer(
            model, config, {"fixture": "tiny-text", "receiver": kind}, rng
        )
        for _ in range(config.steps):
            losses = trainer.step(next_batch("train"))
            assert losses.total.isfinite() and losses.token_count > 0
        trainer.save_resume(tmp_path / f"{kind}-{stage}.pt")
        unchanged = [
            torch.equal(parameter, native_state[name])
            for name, parameter in model.receiver.native.named_parameters()
        ]
        assert all(unchanged) == (stage != "trunk")

    model.eval()
    with torch.no_grad():
        batch = next_batch("selection")
        result = model(
            batch.sender_ids,
            batch.receiver_ids,
            batch.candidate_event_ids,
            batch.frontier,
            hard=False,
        )
    assert result.logits.shape[-1] == table.inventory.outputs
    assert table.inventory.start >= result.logits.shape[-1]
    assert F.cross_entropy(
        result.logits.flatten(0, 1), batch.targets.flatten()
    ).isfinite()
    assert batch.target_bytes.sum() > 0

    modes = [module.training for module in model.modules()]
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    controls = evaluate_controls(
        model, batch, permutation=torch.tensor([1, 0]), hard=False
    )
    assert set(controls) == {"correct", "shuffled", "zero_message", "native"}
    for scores in controls.values():
        assert scores.nll.shape == (2,)
        assert not scores.nll.requires_grad
        assert scores.aggregate()["canonical_path_bpb"] > 0
        torch.testing.assert_close(scores.utf8_bytes, batch.target_bytes)
    torch.testing.assert_close(
        controls["zero_message"].nll, controls["native"].nll, rtol=0, atol=0
    )
    assert modes == [module.training for module in model.modules()]

    model.eval()

    single = replace(
        batch, **{name: getattr(batch, name)[:1] for name in batch.__dataclass_fields__}
    )
    with torch.no_grad():
        expected = model(
            single.sender_ids,
            single.receiver_ids,
            single.candidate_event_ids,
            single.frontier,
            hard=False,
        ).logits[0]
    length = int(single.targets.ne(-100).sum().item())
    first_live = int(single.targets[0].ne(-100).nonzero()[0].item())
    length += first_live
    session = GenerationSession(model, symbols, hard=False)
    actual = torch.stack(
        [session.push(int(i)) for i in single.receiver_ids[0, :length]]
    )
    torch.testing.assert_close(actual, expected[:length], rtol=1e-4, atol=1e-5)
    assert len(session.messages) == int(single.frontier[0, length - 1].item()) + 1

    prompt = [table.pieces[0]]
    primed = GenerationSession(model, symbols, hard=False)
    primed.prime(prompt)
    assert len(primed.messages) == 1 and not primed.partial
    with pytest.raises(SessionLimitError, match="empty-event"):
        primed.push(symbols.inventory.stop)
    with pytest.raises(ValueError, match="START must occur exactly once"):
        primed.push(symbols.inventory.start)

    capped = GenerationSession(model, symbols, hard=False)
    capped.receiver_caches = model.receiver.native.new_caches(1)
    capped.push(symbols.inventory.start)
    with pytest.raises(SessionLimitError, match="receiver-context-limit"):
        capped.push(0)
    assert not capped.partial and capped.receiver_caches[0].pos == 1

    monkeypatch.setattr("execution.generation.sample", lambda *args: 0)
    record = generate_protocol(
        model,
        symbols,
        prompt,
        GenerationLimits(100, 3, 100, 1.0, 1.0),
        seed=42,
        hard=False,
    )
    assert record["censored"] and record["stop_reason"] == "symbol-limit"
    assert record["sender_events"] == 1

    stop = symbols.inventory.stop
    masks = []

    def greedy_word(logits, *args):
        allowed = logits.isfinite()
        masks.append(allowed)
        choices = [i for i in allowed.nonzero().flatten().tolist() if i != stop]
        return choices[0] if choices else stop

    monkeypatch.setattr("execution.generation.sample", greedy_word)
    record = generate_protocol(
        model,
        symbols,
        prompt,
        GenerationLimits(12, 60, 3, 1.0, 1.0),
        seed=42,
        hard=False,
    )
    assert not masks[0][stop]
    assert record["forced_word_breaks"] > 0
    assert any(mask[stop] and int(mask.sum()) == 1 for mask in masks)
    assert record["stop_reason"] in ("byte-budget", "sender-context-limit")
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="no fixed rows"):
        evaluate_controls(model, batch, permutation=torch.tensor([0, 1]), hard=False)

    model.train()
    modes = [module.training for module in model.modules()]
    with pytest.raises(IndexError):
        evaluate_controls(
            model,
            replace(batch, targets=torch.full_like(batch.targets, 99999)),
            permutation=torch.tensor([1, 0]),
            hard=False,
        )
    assert modes == [module.training for module in model.modules()]
