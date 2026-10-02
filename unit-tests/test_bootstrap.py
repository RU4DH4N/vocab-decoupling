from dataclasses import replace

import numpy as np
import torch
from test_protocol_stages import activate, batch, tiny_model

import execution.bootstrap as bootstrap
from data.communication import candidate_events
from execution.bootstrap import _reads, latent_nll
from models.protocol.model import gather_events


def test_latent_path_with_a_flat_prior_is_the_mean_read_likelihood():
    torch.manual_seed(3)
    model = tiny_model().eval()
    activate(model)
    with torch.no_grad():
        for parameter in model.receiver.scorers[0].parameters():
            parameter.zero_()
    rows = batch(np.random.default_rng(0))
    with torch.no_grad():
        memory = model.sender(rows.sender_ids)
        likelihoods, priors = _reads(model, rows, memory)
        nll, live = latent_nll(model, rows, memory)
    allowed = priors.isfinite()
    counts = allowed.sum(-1, keepdim=True)
    torch.testing.assert_close(
        priors.exp().masked_fill(~allowed, 0), allowed / counts, atol=1e-6, rtol=0
    )
    mean = (likelihoods.exp() * allowed).sum(-1) / counts.squeeze(-1)
    torch.testing.assert_close(nll[live], -mean.log()[live], atol=1e-5, rtol=1e-5)


def test_reads_differ_by_event_once_communication_is_active():
    torch.manual_seed(3)
    model = tiny_model().eval()
    activate(model)
    rows = batch(np.random.default_rng(0))
    with torch.no_grad():
        likelihoods, priors = _reads(model, rows, model.sender(rows.sender_ids))
    both = priors[..., :2].isfinite().all(-1)
    assert not torch.allclose(likelihoods[..., 0][both], likelihoods[..., 1][both])


def test_priors_follow_event_ids_through_the_candidate_window():
    torch.manual_seed(5)
    model = tiny_model().eval()
    activate(model)
    with torch.no_grad():
        for parameter in model.receiver.scorers[0].parameters():
            parameter.normal_()
    window = model.receiver.candidate_window
    rows = batch(np.random.default_rng(0))
    rows = replace(rows, candidate_event_ids=candidate_events(rows.frontier, window))
    with torch.no_grad():
        memory = model.sender(rows.sender_ids)
        _, priors = _reads(model, rows, memory)
        full = model.receiver(
            rows.receiver_ids,
            gather_events(memory, rows.candidate_event_ids),
            rows.candidate_event_ids,
            rows.frontier,
            hard=False,
        ).correspondence[0]
    for offset in range(window):
        reachable = rows.frontier.ge(offset)
        torch.testing.assert_close(
            priors[..., offset][reachable], full[..., window - 1 - offset][reachable]
        )
        assert torch.isinf(priors[..., offset][~reachable]).all()


def test_checkpointed_reads_give_the_same_loss_and_gradients(monkeypatch):
    def run():
        torch.manual_seed(5)
        model = tiny_model()
        activate(model)
        rows = batch(np.random.default_rng(0))
        memory = model.sender(rows.sender_ids)
        nll, live = latent_nll(model, rows, memory)
        nll[live].mean().backward()
        return nll.detach(), {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

    loss, grads = run()
    monkeypatch.setattr(
        bootstrap, "checkpoint", lambda fn, *args, use_reentrant: fn(*args)
    )
    expected_loss, expected_grads = run()
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    assert grads.keys() == expected_grads.keys() and grads
    for name, grad in grads.items():
        torch.testing.assert_close(grad, expected_grads[name], rtol=1e-6, atol=1e-7)
