import numpy as np
import pytest
import torch
from test_protocol_stages import activate, batch, tiny_model

from data.communication import ReceiverUnits
from execution.planning_training import oracle_future
from execution.protocol_evaluation import position_scores, prefix_ambiguity
from models.protocol.lookahead import LookaheadReceiver, at_steps
from models.protocol.planner import (
    EventPlanner,
    propose_beams,
    roll_ahead_batch,
    roll_ahead_last,
)
from models.protocol.receiver import ReceiverReadout
from models.shared.symbols import START, STOP


def planner():
    return EventPlanner(8, 8, 16, 4).eval()


def greedy(planning, message, max_symbols):
    hidden = planning.decoder.initial_hidden(1, message).unsqueeze(0)
    previous = torch.tensor([START])
    raw = bytearray()
    for step in range(max_symbols):
        logits, hidden = planning.decoder.step(previous, hidden, message)
        log_probs = logits.float().log_softmax(-1)
        if not step:
            log_probs[:, STOP] = -torch.inf
        symbol = int(log_probs.argmax(-1).item())
        if symbol == STOP:
            return bytes(raw), True
        raw.append(symbol)
        previous = torch.tensor([symbol])
    return bytes(raw), False


def scripted(planning):
    def step(previous, hidden, current):
        logits = torch.full((previous.shape[0], STOP + 1), -10.0)
        started = previous.ne(START)
        logits[:, 97] = torch.where(started, -10.0, current[:, 0])
        logits[:, 98] = torch.where(started, -10.0, current[:, 1])
        logits[:, STOP] = torch.where(started, 0.0, -10.0)
        return logits, hidden

    planning.decoder.step = step
    return planning


def teacher_forced(planning, message, raw):
    symbols = torch.tensor([[START, *raw]])
    targets = torch.tensor([*raw, STOP])
    with torch.no_grad():
        log_probs = planning(message, symbols)[0].float().log_softmax(-1)
    return float(log_probs[torch.arange(len(targets)), targets].sum())


@pytest.mark.parametrize("stop_bias", [0.0, 0.5, 2.0])
def test_single_beam_is_greedy_decoding(stop_bias):
    torch.manual_seed(5)
    planning = planner()
    with torch.no_grad():
        planning.decoder.current_channel.up.weight.normal_(0, 3.0)
        planning.decoder.output.bias[STOP] = stop_bias
    messages = torch.randn(12, 8)
    beams = propose_beams(planning, messages, width=1, max_symbols=6)
    for index, message in enumerate(messages):
        raw, stopped = greedy(planning, message[None], 6)
        assert bool(beams.available[index, 0]) == stopped
        if stopped:
            assert beams.raw[index][0] == raw


def test_beams_are_distinct_sorted_nonempty_and_scored_by_the_planner():
    torch.manual_seed(7)
    planning = planner()
    with torch.no_grad():
        planning.decoder.current_channel.up.weight.normal_(0, 3.0)
        planning.decoder.output.bias[STOP] = 1.0
    messages = torch.randn(6, 8)
    beams = propose_beams(planning, messages, width=4, max_symbols=5)
    assert beams.available.any()
    for index, message in enumerate(messages):
        live = [
            (raw, float(score))
            for raw, score, flag in zip(
                beams.raw[index],
                beams.log_probs[index],
                beams.available[index],
                strict=True,
            )
            if flag
        ]
        assert all(raw for raw, _ in live)
        assert len({raw for raw, _ in live}) == len(live)
        scores = [score for _, score in live]
        assert scores == sorted(scores, reverse=True)
        for raw, score in live:
            assert score == pytest.approx(
                teacher_forced(planning, message[None], raw), abs=1e-4
            )


def test_planner_full_and_incremental_logits_match():
    model = planner()
    messages = torch.randn(2, 8)
    previous = torch.randint(0, 258, (2, 6))
    with torch.no_grad():
        full = model(messages, previous)
        hidden = model.decoder.initial_hidden(2, messages).unsqueeze(0)
        incremental = []
        for symbol in previous.unbind(1):
            logits, hidden = model.decoder.step(symbol, hidden, messages)
            incremental.append(logits)
        torch.testing.assert_close(
            full, torch.stack(incremental, 1), rtol=1e-4, atol=1e-5
        )


def futures(shape, fill):
    return (
        torch.full((*shape, 8), fill),
        torch.full(shape, fill),
        torch.ones(shape, dtype=torch.bool),
    )


def test_future_zero_initialization_is_exact_and_only_future_parameters_learn():
    model = tiny_model().eval()
    activate(model)
    data = batch(np.random.default_rng(23))
    memory = torch.randn(2, 5, 4, 8)
    future = torch.randn(2, 5, 2, 3, 8)
    log_probs = torch.randn(2, 5, 2, 3)
    available = torch.ones(2, 5, 2, 3, dtype=torch.bool)
    receiver = LookaheadReceiver(model.receiver, 2, 4).train()
    current = model.receiver(
        data.receiver_ids, memory, data.candidate_event_ids, data.frontier, hard=False
    ).logits
    result = receiver(
        data.receiver_ids,
        memory,
        data.candidate_event_ids,
        data.frontier,
        future,
        log_probs,
        available,
        torch.arange(5).expand(2, 5),
        hard=False,
    ).logits
    torch.testing.assert_close(result, current, rtol=0, atol=0)
    result.square().mean().backward()
    assert all(p.grad is None for p in receiver.current.parameters())
    assert any(
        p.grad is not None and p.grad.any() for p in receiver.channels.parameters()
    )
    assert not receiver.current.training


def test_unavailable_future_is_exact_noop_even_after_training():
    model = tiny_model().eval()
    receiver = LookaheadReceiver(model.receiver, 1, 4).eval()
    with torch.no_grad():
        for layer in receiver.channels:
            layer[0].up.weight.normal_()
    data = batch(np.random.default_rng(23))
    memory = torch.randn(2, 5, 4, 8)
    future, log_probs, available = futures((2, 5, 1, 3), torch.nan)
    with torch.no_grad():
        expected = model.receiver(
            data.receiver_ids,
            memory,
            data.candidate_event_ids,
            data.frontier,
            hard=False,
        ).logits
        actual = receiver(
            data.receiver_ids,
            memory,
            data.candidate_event_ids,
            data.frontier,
            future,
            log_probs,
            ~available,
            torch.arange(5).expand(2, 5),
            hard=False,
        ).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_event_futures_match_futures_copied_to_every_step():
    model = tiny_model().eval()
    receiver = LookaheadReceiver(model.receiver, 2, 4).eval()
    with torch.no_grad():
        for layer in receiver.channels:
            for channel in layer:
                channel.up.weight.normal_()
    data = batch(np.random.default_rng(29))
    torch.manual_seed(3)
    steps = data.receiver_ids.shape[1]
    memory = torch.randn(2, steps, 4, 8)
    future = torch.randn(2, 6, 2, 3, 8)
    log_probs = torch.randn(2, 6, 2, 3)
    available = torch.rand(2, 6, 2, 3) > 0.3
    positions = torch.randint(-1, 6, (2, steps))
    inputs = (data.receiver_ids, memory, data.candidate_event_ids, data.frontier)
    with torch.no_grad():
        per_event = receiver(
            *inputs, future, log_probs, available, positions, hard=False
        ).logits
        per_step = receiver(
            *inputs,
            at_steps(future, positions),
            at_steps(log_probs, positions),
            at_steps(available, positions),
            torch.arange(steps).expand(2, steps),
            hard=False,
        ).logits
    torch.testing.assert_close(per_event, per_step, rtol=1e-5, atol=1e-5)


def test_rollout_rejects_future_beyond_context():
    model = tiny_model().eval()
    codes = model.sender.trunk_table[torch.zeros(1, 8, dtype=torch.long)]
    with pytest.raises(ValueError, match="speculative horizon"):
        roll_ahead_batch(
            model.sender, planner(), codes, horizon=1, width=2, max_symbols=2
        )


@pytest.mark.parametrize("horizon", [1, 2])
def test_rollout_messages_are_the_sender_after_each_hypothesis(horizon):
    torch.manual_seed(5)
    model = tiny_model().eval()
    planning = scripted(planner())
    vocab = ["a", " b", " c", " d"]
    ids = torch.randint(0, len(vocab), (2, 8 - horizon))
    plan = roll_ahead_batch(
        model.sender,
        planning,
        model.sender.trunk_table[ids],
        horizon=horizon,
        width=2,
        max_symbols=4,
    )
    assert plan.available.all()
    for row in range(ids.shape[0]):
        for event in range(ids.shape[1]):
            prefix = [vocab[i].encode() for i in ids[row, : event + 1].tolist()]
            first = plan.proposals[row * ids.shape[1] + event]
            assert set(first) == {b"a", b"b"}
            for hypothesis, word in enumerate(first):
                history = [*prefix, word]
                for step in range(horizon):
                    codes, _ = model.sender.encode_trunk_vocab(history)
                    with torch.no_grad():
                        expected = model.sender.state(codes[None])[0, -1]
                    torch.testing.assert_close(
                        plan.messages[row, event, step, hypothesis],
                        expected,
                        atol=1e-5,
                        rtol=1e-4,
                    )
                    with torch.no_grad():
                        following = propose_beams(
                            planning, expected[None], width=1, max_symbols=4
                        )
                    history.append(following.raw[0][0])


@pytest.mark.parametrize("horizon", [1, 2])
def test_last_position_plan_from_caches_matches_the_full_rollout(horizon):
    torch.manual_seed(11)
    model = tiny_model().eval()
    planning = planner()
    with torch.no_grad():
        planning.decoder.current_channel.up.weight.normal_(0, 3.0)
    ids = torch.randint(0, 4, (1, 8 - horizon))
    codes = model.sender.trunk_table[ids]
    full = roll_ahead_batch(
        model.sender, planning, codes, horizon=horizon, width=3, max_symbols=4
    )
    caches = model.sender.new_caches(model.sender.trunk.config.max_seq_len)
    with torch.no_grad():
        memory = model.sender.state(codes, caches)
    last = roll_ahead_last(
        model.sender,
        planning,
        caches,
        memory[:, -1:],
        horizon=horizon,
        width=3,
        max_symbols=4,
    )
    torch.testing.assert_close(last.messages, full.messages[0, -1])
    torch.testing.assert_close(last.log_probs, full.log_probs[0, -1])
    assert torch.equal(last.available, full.available[0, -1])


def test_oracle_futures_are_the_true_next_events():

    model = tiny_model().eval()
    rows = torch.randint(0, 4, (2, 6))
    oracle = oracle_future(model, rows, 2, torch.device("cpu"))
    with torch.no_grad():
        full = model.sender(rows)
    events = rows.shape[1] - 1
    for step in range(2):
        for event in range(events):
            reachable = event + 1 + step <= events
            assert bool(oracle["available"][0, event, step, 0]) == reachable
            if reachable:
                torch.testing.assert_close(
                    oracle["future"][:, event, step, 0], full[:, event + 1 + step]
                )


def test_position_scores_bin_symbols_by_place_in_their_event():

    table = ReceiverUnits.bytes(["ab", " c", "d"])
    batch = table.batch(
        torch.tensor([[2, 0, 1]]),
        device="cpu",
        max_receiver_steps=16,
        candidate_window=3,
    )
    logits = torch.zeros(*batch.targets.shape, table.inventory.outputs)
    totals = position_scores(
        ReceiverReadout(logits, ()), batch, table.inventory, width=3
    )
    assert totals[1].tolist() == [2.0, 2.0, 2.0]
    uniform = torch.log(torch.tensor(float(table.inventory.outputs))).item()
    torch.testing.assert_close(totals[0], totals[1] * uniform, rtol=1e-6, atol=1e-6)


def test_prefix_ambiguity_counts_units_sharing_each_prefix():

    table = prefix_ambiguity(["ab", "ac", "b"], 3)
    assert table.tolist() == [[3, 2, 1], [3, 2, 1], [3, 1, 0]]
