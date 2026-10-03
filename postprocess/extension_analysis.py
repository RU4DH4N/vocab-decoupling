import argparse
import json
import math
from pathlib import Path

from tokenizers import Tokenizer

from claims.receiver_scaling import POINTS
from claims.receiver_vocabulary import VOCABULARIES
from execution.design import _body_parameters
from execution.receiver_tokenizers import tokenizer_path
from framework.checkpoints import write_json
from postprocess.analysis import Run, difference, interval, number, rows, table

BYTES = 256


def measured(root: Path, split: str, extension: str | None, variant: str) -> Path:
    base = root / "seed-0"
    if extension is not None:
        base = base / "extensions" / extension
    return (base if split == "selection" else base / "reporting") / variant


def load(folder: Path) -> tuple[dict, dict]:
    evaluation = json.loads((folder / "evaluation.json").read_text())
    generation = json.loads((folder / "generation.json").read_text())
    return evaluation, generation


def paired(run: Run, evaluation: dict, reference: dict, arm: str, other: str) -> dict:
    if evaluation["group_ids"] != reference["group_ids"]:
        raise ValueError("paired receivers were not scored on the same windows")
    left, utf8 = rows(evaluation, arm)
    right, _ = rows(reference, other)
    return difference(run, [left], [right], [utf8])


def bits(evaluation: dict) -> float:
    nats, utf8 = rows(evaluation, "correct")
    return float(nats.sum() / (math.log(2) * utf8.sum()))


def scaling_point(
    root: Path, split: str, design: dict, point: int | str
) -> tuple[str, dict, dict, dict]:
    if isinstance(point, str):
        evaluation, generation = load(measured(root, split, None, "fresh-byte"))
        return "design", design["shared"]["native"], evaluation, generation
    label = f"bytes-{point}"
    evaluation, generation = load(measured(root, split, label, "fresh-byte"))
    return label, evaluation["receiver"]["native"], evaluation, generation


def scaling(root: Path, split: str) -> dict:
    run = Run(root, split)
    design = json.loads((root / "design.json").read_text())
    trunk = run.inputs["trunk_parameters"]
    points = [scaling_point(root, split, design, p) for p in POINTS[trunk]]
    points.sort(key=lambda point: _body_parameters(point[1]))
    largest = points[-1][2]
    receivers = []
    for label, native, evaluation, generation in points:
        receivers.append(
            {
                "receiver": label,
                "shape": f"{native['d_model']}x{native['n_layers']}",
                "body_parameters": _body_parameters(native),
                "correct_bits_per_byte": bits(evaluation),
                "minus_largest": paired(run, evaluation, largest, "correct", "correct"),
                "shuffled_minus_correct": paired(
                    run, evaluation, evaluation, "shuffled", "correct"
                ),
                "latency_seconds_per_byte": generation["latency_seconds_per_byte"],
            }
        )
    return {
        "output": root.as_posix(),
        "trunk_parameters": design["parameters"]["sender"],
        "receivers": receivers,
    }


def vocabulary_point(root: Path, split: str, label: str) -> tuple[int, dict, dict, str]:
    if label == "bytes":
        evaluation, generation = load(measured(root, split, None, "fresh-byte"))
        return BYTES, evaluation, generation, "v1"
    if label == "bpe-design":
        evaluation, generation = load(measured(root, split, None, "fresh-bpe"))
        size = Tokenizer.from_file(
            str(root / "receiver-tokenizer.json")
        ).get_vocab_size()
        return size, evaluation, generation, "v1"
    vocabulary = int(label.removeprefix("bpe-"))
    evaluation, generation = load(measured(root, split, label, "fresh-bpe"))
    size = Tokenizer.from_file(str(tokenizer_path(root, vocabulary))).get_vocab_size()
    return size, evaluation, generation, "v1.1"


def vocabulary(root: Path, split: str) -> dict:
    run = Run(root, split)
    labels = ["bytes", *(f"bpe-{v}" for v in VOCABULARIES), "bpe-design"]
    points = {label: vocabulary_point(root, split, label) for label in labels}
    reference_bytes = points["bytes"][1]
    reference_bpe = points["bpe-design"][1]
    receivers = []
    for label, (size, evaluation, generation, source) in points.items():
        receivers.append(
            {
                "receiver": label,
                "vocabulary": size,
                "correct_bits_per_byte": bits(evaluation),
                "minus_bytes": paired(
                    run, evaluation, reference_bytes, "correct", "correct"
                ),
                "minus_design_bpe": paired(
                    run, evaluation, reference_bpe, "correct", "correct"
                ),
                "shuffled_minus_correct": paired(
                    run, evaluation, evaluation, "shuffled", "correct"
                ),
                "latency_seconds_per_byte": generation["latency_seconds_per_byte"],
                "bytes_per_second": generation["bytes_per_second"],
                "measured_in": source,
            }
        )
    return {"output": root.as_posix(), "receivers": receivers}


def scaling_section(results: list[dict]) -> list[str]:
    body = [
        [
            f"{trunk['trunk_parameters']:,}",
            receiver["receiver"],
            receiver["shape"],
            f"{receiver['body_parameters']:,}",
            number(receiver["correct_bits_per_byte"]),
            interval(receiver["minus_largest"]),
            interval(receiver["shuffled_minus_correct"]),
        ]
        for trunk in results
        for receiver in trunk["receivers"]
    ]
    header = [
        "trunk parameters",
        "receiver",
        "shape",
        "body parameters",
        "correct bpb",
        "receiver − largest",
        "shuffled − correct",
    ]
    return ["## C8", table(header, body)]


def vocabulary_section(result: dict) -> list[str]:
    body = [
        [
            receiver["receiver"],
            f"{receiver['vocabulary']:,}",
            number(receiver["correct_bits_per_byte"]),
            interval(receiver["minus_bytes"]),
            interval(receiver["minus_design_bpe"]),
            interval(receiver["shuffled_minus_correct"]),
            f"{receiver['latency_seconds_per_byte']['p50']:.5f}",
            receiver["measured_in"],
        ]
        for receiver in result["receivers"]
    ]
    header = [
        "receiver",
        "vocabulary",
        "correct bpb",
        "receiver − bytes",
        "receiver − design BPE",
        "shuffled − correct",
        "latency p50 (s/byte)",
        "measured in",
    ]
    return ["## C9", table(header, body)]


def analyse(
    scaling_roots: list[Path], vocabulary_root: Path, split: str
) -> tuple[dict, str]:
    c8 = [scaling(root, split) for root in scaling_roots]
    c9 = vocabulary(vocabulary_root, split)
    sections = [
        f"# Extensions ({split} split, 1 seed, 95% paired bootstrap over windows)",
        "Differences are in bits per byte; positive means the second arm is better.",
        *scaling_section(c8),
        *vocabulary_section(c9),
    ]
    return {"split": split, "C8": c8, "C9": c9}, "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling", type=Path, nargs="+", required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "reporting"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    results, markdown = analyse(args.scaling, args.vocabulary, args.split)
    write_json(args.report / f"extensions-{args.split}.json", results)
    (args.report / f"extensions-{args.split}.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
