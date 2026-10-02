from dataclasses import replace

import numpy as np
import pytest
import torch

from data.communication import ProtocolBatch
from execution.protocol_stages import StageConfig, StageTrainer
from framework.resume import current, load
from models.bpe.bpe import BPEModel
from models.protocol.model import ProtocolModel, gather_events
from models.protocol.receiver import Receiver
from models.protocol.sender import CoarseSender


def tiny_model():
    torch.manual_seed(17)
    sender = CoarseSender(
        in_dims=512,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2,
        d_meaning=8,
        codebook_seed=3,
        dropout=0.2,
        multiple_of=8,
        max_seq_len=8,
        rope_theta=10000,
        norm_eps=1e-5,
        initialiser_range=0.02,
    )
    sender.build_table(["a", " b", " c", " d"])
    native = BPEModel(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2,
        dropout=0.2,
        multiple_of=8,
        max_seq_len=16,
        rope_theta=10000,
        norm_eps=1e-5,
        initialiser_range=0.02,
        tie_word_embeddings=True,
    )
    model = ProtocolModel(sender, Receiver(native, 8, 4, 4, 3, output_symbols=32))
    return model


def batch(rng):
    return ProtocolBatch(
        sender_ids=torch.tensor(rng.integers(0, 4, (2, 4))),
        receiver_ids=torch.randint(0, 32, (2, 5)),
        targets=torch.randint(0, 32, (2, 5)),
        candidate_event_ids=torch.arange(4).expand(2, 5, 4),
        frontier=torch.tensor([[0, 0, 1, 2, 3]]).expand(2, 5),
        alignment_targets=torch.tensor([[0, 0, 1, 2, 3]]).expand(2, 5),
        target_bytes=torch.full((2,), 5),
    )


def config(stage):
    return StageConfig(
        name=stage,
        steps=4,
        learning_rate=0.002,
        weight_decay=0.01,
        alignment_weight=1.0 if stage == "alignment" else 0.0,
        grad_clip=1.0,
        warmup_steps=1,
        precision="fp32",
    )


def activate(model):
    with torch.no_grad():
        for channel in model.receiver.channels:
            channel.up.weight.normal_(std=0.1)


@pytest.mark.parametrize(
    "stage,trainable",
    [
        ("alignment", {"scorers"}),
        ("communication", {"channels"}),
        ("joint", {"scorers", "channels"}),
        ("trunk", {"sender", "native", "scorers", "channels"}),
    ],
)
def test_only_declared_stage_parameters_can_change(stage, trainable):
    model = tiny_model()
    activate(model)
    trainer = StageTrainer(
        model, config(stage), {"test": "stage-isolation"}, np.random.default_rng(23)
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    losses = trainer.step(batch(trainer.rng))
    assert losses.total.isfinite()
    changed_groups = set()
    for name, p in model.named_parameters():
        group = "sender" if name.startswith("sender.") else name.split(".")[1]
        if group not in trainable:
            assert not p.requires_grad and p.grad is None
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        elif not torch.equal(p, before[name]):
            changed_groups.add(group)
    assert changed_groups == trainable
    assert not model.receiver.native.training
    assert model.sender.training == (stage == "trunk")


def test_stage_transition_clears_old_gradients_and_invalidates_previous_trainer():
    model = tiny_model()
    first = StageTrainer(
        model, config("alignment"), {"test": "stage"}, np.random.default_rng(23)
    )
    first.step(batch(first.rng))
    assert any(p.grad is not None for p in model.receiver.scorers.parameters())
    second = StageTrainer(model, config("communication"), {"test": "stage"}, first.rng)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(RuntimeError, match="stage changed"):
        first.step(batch(first.rng))
    second.step(batch(second.rng))


def test_task_only_objective_does_not_depend_on_alignment_labels():
    model = tiny_model()
    activate(model)
    trainer = StageTrainer(
        model, config("joint"), {"test": "no-labels"}, np.random.default_rng(23)
    )
    model.eval()
    data = batch(trainer.rng)
    ordinary = trainer.objective(data)
    impossible = trainer.objective(
        replace(data, alignment_targets=torch.full_like(data.alignment_targets, 999))
    )
    torch.testing.assert_close(ordinary.total, impossible.total, rtol=0, atol=0)
    assert impossible.alignment.isinf()


def test_invalid_alignment_cannot_take_an_optimizer_step():
    model = tiny_model()
    trainer = StageTrainer(
        model, config("alignment"), {"test": "bad-alignment"}, np.random.default_rng(23)
    )
    data = batch(trainer.rng)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(FloatingPointError, match="optimizer step was not taken"):
        trainer.step(
            replace(
                data, alignment_targets=torch.full_like(data.alignment_targets, 999)
            )
        )
    assert trainer.completed == 0
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_interrupted_trunk_stage_resumes_bit_exactly(tmp_path):
    stage = config("trunk")

    def trainer():
        model = tiny_model()
        activate(model)
        return StageTrainer(
            model,
            stage,
            {"data-sha": "synthetic", "source-sha": "fixture"},
            np.random.default_rng(23),
        )

    uninterrupted = trainer()
    for _ in range(stage.steps):
        uninterrupted.step(batch(uninterrupted.rng))
    interrupted = trainer()
    for _ in range(2):
        interrupted.step(batch(interrupted.rng))
    path = tmp_path / "stage.resume.pt"
    interrupted.save_resume(path)
    resumed = trainer()
    resumed.resume(load(path, resumed.fingerprint))
    assert resumed.completed == 2
    for _ in range(resumed.completed, stage.steps):
        resumed.step(batch(resumed.rng))
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[name], rtol=0, atol=0
        )
    assert uninterrupted.rng.bit_generator.state == resumed.rng.bit_generator.state
    assert resumed.completed == stage.steps
    with pytest.raises(RuntimeError, match="already complete"):
        resumed.step(batch(resumed.rng))


def test_resume_rejects_changed_stage_or_provenance(tmp_path):
    trainer = StageTrainer(
        tiny_model(), config("alignment"), {"data": "one"}, np.random.default_rng(23)
    )
    path = tmp_path / "stage.resume.pt"
    trainer.save_resume(path)
    changed = StageTrainer(
        tiny_model(), config("joint"), {"data": "one"}, np.random.default_rng(23)
    )
    with pytest.raises(ValueError, match="different training configuration"):
        changed.resume(load(path, changed.fingerprint))
    changed_data = StageTrainer(
        tiny_model(), config("alignment"), {"data": "two"}, np.random.default_rng(23)
    )
    with pytest.raises(ValueError, match="different training configuration"):
        changed_data.resume(load(path, changed_data.fingerprint))


def test_gather_duplicates_events_and_padding_has_no_gradient():
    memory = torch.randn(1, 3, 4, requires_grad=True)
    ids = torch.tensor([[[1, 1, -1]]])
    selected = gather_events(memory, ids)
    torch.testing.assert_close(selected[0, 0, 0], selected[0, 0, 1])
    assert torch.count_nonzero(selected[0, 0, 2]) == 0
    selected.sum().backward()
    assert memory.grad is not None
    torch.testing.assert_close(memory.grad[0, 1], torch.full((4,), 2.0))
    assert torch.count_nonzero(memory.grad[0, 0]) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 0),
        ("steps", True),
        ("warmup_steps", 4),
        ("learning_rate", float("nan")),
        ("alignment_weight", -1),
        ("precision", "fp16"),
    ],
)
def test_invalid_stage_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        replace(config("joint"), **{field: value})


def test_stale_resume_state_is_discarded_and_training_starts_over(tmp_path):
    trainer = StageTrainer(
        tiny_model(), config("alignment"), {"data": "one"}, np.random.default_rng(23)
    )
    path = tmp_path / "stage.resume.pt"
    trainer.save_resume(path)
    assert current(path, trainer.fingerprint) is not None
    changed = StageTrainer(
        tiny_model(), config("alignment"), {"data": "two"}, np.random.default_rng(23)
    )
    assert current(path, changed.fingerprint) is None
    assert not path.exists()
    assert current(path, trainer.fingerprint) is None
