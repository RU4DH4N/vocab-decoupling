import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from data.communication import ProtocolBatch, ReceiverUnits
from data.protocol_corpus import EventCorpus, selection_windows
from execution.generation import SymbolBytes, generate_protocol
from execution.layout import Layout
from execution.mauve import mauve_score
from execution.planning_training import (
    future_inputs,
    load_planner,
    lookahead_path,
    make_plan_batch,
    oracle_future,
    primary_path,
)
from execution.protocol_evaluation import (
    RowScores,
    ambiguity_scores,
    position_scores,
    prefix_ambiguity,
    score_rows,
)
from execution.protocol_io import check_inventory, evaluation_model, load_protocol
from execution.sampling import GenerationLimits
from execution.streams import (
    Shard,
    device_streams,
    generate_streams,
    prompt_count,
    summary,
)
from execution.training import encoded
from execution.validation import validate
from execution.windows import protocol_batch
from framework.checkpoints import (
    file_sha256,
    load_checkpoint,
    read_sidecar,
    write_json,
)
from framework.runtime import (
    device_synchronize,
)
from models.protocol.lookahead import LookaheadReceiver
from models.protocol.model import ProtocolModel, gather_events
from models.protocol.receiver import Receiver, ReceiverReadout

CONDITIONS = ("real", "shuffled", "repeated", "zero", "oracle", "oracle_fitted")


def _generation(config: dict, layout: Layout, device: torch.device, split: str) -> None:
    generation = config["generation"]
    records, timed = generate_streams(
        generate_lookahead,
        (layout.shared, layout.replicate, split, str(device)),
        prompt_count(layout.evaluation_prompts(split)),
        generation["latency_samples"],
        device_streams(generation, device),
    )
    write_json(
        layout.measured(split) / "planning/generation.json",
        {
            "checkpoint_sha256": file_sha256(layout.run / "planning/lookahead.pt"),
            **summary(records, timed, layout.evaluation_prompts(split), generation),
        },
    )


def _check_adapters(
    config: dict, layout: Layout, corpus: EventCorpus, split: str
) -> None:
    primary = primary_path(config, layout)
    check_inventory(primary, corpus, split)
    for source in ("planner", "oracle"):
        metadata = read_sidecar(lookahead_path(layout, source))
        if metadata["primary_sha256"] != file_sha256(primary):
            raise ValueError("future adapter belongs to another primary receiver")


def _receivers(
    config: dict,
    layout: Layout,
    model: ProtocolModel[Receiver],
    device: torch.device,
) -> dict[str, LookaheadReceiver]:
    options = config["lookahead"]
    receivers = {}
    for source in ("planner", "oracle"):
        payload = load_checkpoint(lookahead_path(layout, source), map_location=device)
        receiver = LookaheadReceiver(
            model.receiver, options["horizon"], options["rank"]
        ).to(device)
        receiver.channels.load_state_dict(payload["model"], strict=True)
        receivers[source] = receiver.eval()
    return receivers


def _tally_proposals(
    proposals: dict[str, int], record: dict, truth: list[bytes]
) -> None:
    live = record["available"][:, :, 0].flatten(0, 1).tolist()
    for guesses, actual, flags in zip(record["proposals"], truth, live, strict=True):
        proposals["positions"] += 1
        proposals["available"] += any(flags)
        proposals["top"] += flags[0] and guesses[0] == actual
        proposals["within"] += any(
            flag and guess == actual for guess, flag in zip(guesses, flags, strict=True)
        )


class _Tally:
    def __init__(
        self,
        table: ReceiverUnits,
        vocab: list[str],
        width: int,
        device: torch.device,
    ) -> None:
        self.table, self.width = table, width
        self.ambiguity = torch.from_numpy(prefix_ambiguity(vocab, width))
        self.decades = int(np.log10(len(vocab))) + 1
        self.rows: dict[str, list[RowScores]] = {k: [] for k in CONDITIONS}
        self.positions = {
            k: torch.zeros(2, width, dtype=torch.float64, device=device)
            for k in CONDITIONS
        }
        self.ambiguous = {
            k: torch.zeros(2, self.decades, dtype=torch.float64, device=device)
            for k in CONDITIONS
        }

    def add(
        self,
        condition: str,
        result: ReceiverReadout,
        batch: ProtocolBatch,
        rows: Tensor,
    ) -> None:
        inventory = self.table.inventory
        self.rows[condition].append(score_rows(result, batch))
        self.positions[condition] += position_scores(
            result, batch, inventory, self.width
        )
        self.ambiguous[condition] += ambiguity_scores(
            result, batch, rows, self.ambiguity, inventory, self.decades
        )

    def report(self) -> dict:
        scores = {k: RowScores.concatenate(v) for k, v in self.rows.items()}
        return {
            "scores": {key: value.aggregate() for key, value in scores.items()},
            "paired_nll": {key: value.nll.tolist() for key, value in scores.items()},
            "utf8_bytes": scores["real"].utf8_bytes.tolist(),
            "by_position": {
                key: {"nll": value[0].tolist(), "tokens": value[1].tolist()}
                for key, value in self.positions.items()
            },
            "by_prefix_ambiguity": {
                key: {"nll": value[0].tolist(), "tokens": value[1].tolist()}
                for key, value in self.ambiguous.items()
            },
        }


def _score_conditions(
    receivers: dict[str, LookaheadReceiver],
    record: dict,
    batch: ProtocolBatch,
    tally: _Tally,
) -> None:
    candidates = gather_events(record["memory"], batch.candidate_event_ids)
    for condition in CONDITIONS:
        fitted = condition == "oracle_fitted"
        future, log_probs, available = future_inputs(
            record, "oracle" if fitted else condition
        )
        result = receivers["oracle" if fitted else "planner"](
            batch.receiver_ids,
            candidates,
            batch.candidate_event_ids,
            batch.frontier,
            future,
            log_probs,
            available,
            batch.frontier,
            hard=False,
        )
        tally.add(condition, result, batch, record["rows"])


@torch.no_grad()
def measure_future(
    config: dict, layout: Layout, device: torch.device, split: str
) -> None:
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    _check_adapters(config, layout, corpus, split)
    _generation(config, layout, device, split)
    model = evaluation_model(primary_path(config, layout), corpus, split, device)
    receivers = _receivers(config, layout, model, device)
    planner = load_planner(config, layout, device)
    table = encoded(config, layout, "primary", corpus)
    tally = _Tally(table, corpus.vocab, config["corpus"]["max_unit_bytes"] + 1, device)
    rng = np.random.default_rng(config["evaluation_seed"])
    rollout = 0.0
    proposals = {"positions": 0, "available": 0, "top": 0, "within": 0}
    for _ in range(layout.evaluation_batches(config, split)):
        rows, _ = selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="selection"
        )
        device_synchronize(device)
        planned = time.perf_counter()
        record = make_plan_batch(config, model, planner, rows, device)
        device_synchronize(device)
        rollout += time.perf_counter() - planned
        truth = [unit.encode() for units in corpus.decode(rows) for unit in units[1:]]
        _tally_proposals(proposals, record, truth)
        record["oracle"] = oracle_future(
            model, rows, config["lookahead"]["horizon"], device
        )
        batch = protocol_batch(config, table, record["rows"], device)
        _score_conditions(receivers, record, batch, tally)
    write_json(
        layout.measured(split) / "planning/evaluation.json",
        {
            **tally.report(),
            "ambiguity": "decade of the number of inventory units sharing the event's bytes so far",
            "likelihood_domain": "canonical-path",
            "split": split,
            "controls": "frozen-adapter interventions",
            "oracle": "teacher-forced reference futures through the planner-trained channel",
            "oracle_fitted": "teacher-forced futures through a channel trained on them; the interface ceiling, never a deployed arm",
            "rollout_seconds": rollout,
            "planner": {
                "availability": proposals["available"] / proposals["positions"],
                "top_hypothesis_exact": proposals["top"] / proposals["positions"],
                "any_hypothesis_exact": proposals["within"] / proposals["positions"],
                "hypotheses": config["lookahead"]["hypotheses"],
            },
        },
    )


@torch.no_grad()
def generate_lookahead(
    shared: Path, replicate: int, split: str, device: str, indices: list[int]
) -> Shard:
    layout = Layout(shared, replicate)
    config = layout.config()
    target = torch.device(device)
    options = config["lookahead"]
    model, _ = load_protocol(primary_path(config, layout), target, table=False)
    receiver = (
        LookaheadReceiver(model.receiver, options["horizon"], options["rank"])
        .to(target)
        .eval()
    )
    payload = load_checkpoint(lookahead_path(layout, "planner"), map_location=target)
    receiver.channels.load_state_dict(payload["model"], strict=True)
    planner = load_planner(config, layout, target)
    generation = config["generation"]
    limits = GenerationLimits(
        **{
            k: generation[k]
            for k in ("bytes", "symbols", "event_bytes", "temperature", "top_p")
        }
    )
    examples = json.loads(layout.evaluation_prompts(split).read_text())
    records = []
    for index in indices:
        example = examples[index]
        generated = generate_protocol(
            model,
            SymbolBytes.bytes(),
            [list(unit.encode()) for unit in example["prompt_units"]],
            limits,
            seed=config["evaluation_seed"] + index,
            hard=False,
            lookahead=receiver,
            planner=planner,
            planner_max_symbols=config["planner"]["max_symbols"],
            planner_width=options["hypotheses"],
        )
        records.append((index, {**example, **generated}))
        print(f"lookahead generation={index + 1}/{len(examples)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("measure", "mauve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("selection", "reporting"), required=True)
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    config = layout.config()
    validate(config)
    device = torch.device(args.device)
    if args.task == "measure":
        measure_future(config, layout, device, args.split)
    else:
        folder = layout.measured(args.split) / "planning"
        mauve_score(
            folder / "generation.json", config["mauve"], folder / "mauve.json", device
        )


if __name__ == "__main__":
    main()
