import math

from execution.design import PRIMARY_STAGES, RECEIVERS, REPLACEMENT_STAGES
from execution.protocol_stages import StageConfig
from execution.sampling import GenerationLimits


def _schedule(name: str, options: dict) -> None:
    for key in ("steps", "warmup_steps"):
        if type(options[key]) is not int:
            raise ValueError(f"{name}.{key} must be an integer")
    if not 0 <= options["warmup_steps"] < options["steps"]:
        raise ValueError(f"invalid {name} training schedule")
    rate = options["learning_rate"]
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError(f"invalid {name}.learning_rate")
    if not math.isfinite(options["grad_clip"]) or options["grad_clip"] <= 0:
        raise ValueError(f"invalid {name}.grad_clip")
    if not math.isfinite(options["weight_decay"]) or options["weight_decay"] < 0:
        raise ValueError(f"invalid {name}.weight_decay")


def validate(config: dict) -> None:
    for key in (
        "batch_size",
        "events",
        "checkpoint_interval",
        "evaluation_batches",
        "selection_batches",
        "receiver_steps",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["batch_size"] < 2:
        raise ValueError("shuffled controls require at least two examples")
    unit = config["corpus"]["max_unit_bytes"]
    if (
        config["events"] + config["lookahead"]["horizon"]
        > config["sender"]["max_seq_len"]
    ):
        raise ValueError("sender context must include the speculative horizon")
    if (config["events"] + 1) * (unit + 1) > config["receiver_steps"]:
        raise ValueError("receiver steps must fit the entire window including STOPs")
    if config["receiver_steps"] > config["native"]["max_seq_len"]:
        raise ValueError("native receiver context is shorter than the window")
    if config["interface"]["sender_dimensions"] != config["sender"]["d_meaning"]:
        raise ValueError("sender and interface widths differ")
    if (
        config["planner"]["model"]["message_dimensions"]
        != config["sender"]["d_meaning"]
    ):
        raise ValueError("planner width differs from sender")
    hypotheses = config["lookahead"]["hypotheses"]
    if type(hypotheses) is not int or hypotheses < 1:
        raise ValueError("lookahead.hypotheses must be a positive integer")
    if tuple(s["name"] for s in config["stages"]) != PRIMARY_STAGES:
        raise ValueError(f"primary stages must be {PRIMARY_STAGES}")
    if tuple(s["name"] for s in config["replacement_stages"]) != REPLACEMENT_STAGES:
        raise ValueError(f"replacement stages must be {REPLACEMENT_STAGES}")
    for stage in (*config["stages"], *config["replacement_stages"]):
        if stage["learning_rate"] is not None and stage["alignment_weight"] is not None:
            StageConfig(**stage)
    if tuple(config["receivers"]) != RECEIVERS:
        raise ValueError(f"receivers must be {RECEIVERS}")
    seeds = [v["seed"] for v in config["receivers"].values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("native receivers require distinct seeds")
    for section in (
        "native_training",
        "baseline",
        "planner",
        "lookahead",
        "control_training",
    ):
        _schedule(section, config[section])
    options = config["generation"]
    GenerationLimits(
        **{
            k: options[k]
            for k in ("bytes", "symbols", "event_bytes", "temperature", "top_p")
        }
    )
    if 1 + options["prompt_events"] * (unit + 1) > config["native"]["max_seq_len"]:
        raise ValueError("receiver cannot fit the entire generation prompt")
    baseline_context = config["baseline"]["model"]["max_seq_len"]
    if (config["events"] + 1) * unit - 1 > baseline_context:
        raise ValueError("BPE context must fit even an unmerged byte-level window")
    if options["prompt_events"] * unit >= baseline_context:
        raise ValueError("BPE context must fit the prompt and a generation step")
