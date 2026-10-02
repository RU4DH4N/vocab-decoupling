import argparse
import json
import sys
from pathlib import Path

from execution.design import check_inputs
from orchestrator.context import load_context
from orchestrator.contract import Contract
from orchestrator.discovery import discover, script_name
from orchestrator.runner import DependencyError, Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def _parser(default: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if default is None:
        parser.add_argument(
            "contracts",
            nargs="+",
            help="e.g. claims/active.py claims/planning_ahead.py",
        )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--reporting-file", type=Path)
    parser.add_argument(
        "--split",
        choices=("selection", "reporting"),
        required=True,
        help="where to measure; evaluate reporting once, after the design is final",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "cuda:0", "cuda:1"), default="cpu"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--slots",
        type=int,
        default=1,
        help="accelerator capacity shared by steps in proportion to their weight",
    )
    parser.add_argument(
        "--run-name", default="contracts", help="remote progress stream name"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--plan", action="store_true", help="inspect only; never execute"
    )
    modes.add_argument("--run-missing", action="store_true")
    modes.add_argument("--require-complete", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument(
        "--export-graph",
        type=Path,
        help="export data only; render separately with postprocess.graph",
    )
    return parser


def _choose_config(parser: argparse.ArgumentParser) -> Path:
    if not sys.stdin.isatty():
        parser.error("choose --config explicitly outside an interactive terminal")
    choices = sorted((ROOT / "config").glob("*.json"))
    for index, path in enumerate(choices, 1):
        print(f"{index}: {path.name}")
    try:
        choice = int(input("Choose configuration: "))
    except ValueError:
        parser.error("invalid configuration choice")
    if not 1 <= choice <= len(choices):
        parser.error("invalid configuration choice")
    return choices[choice - 1]


def _mode(args: argparse.Namespace) -> str:
    if args.run_missing:
        return "run-missing"
    if args.require_complete:
        return "require-complete"
    return "prompt"


def _show_results(roots: list[Contract]) -> None:
    for root in roots:
        for result in root.outputs():
            print(f"\n{root.label}: {result}")
            if result.suffix == ".json" and result.stat().st_size <= 32768:
                print(json.dumps(json.loads(result.read_text()), indent=2))


def _execute(args: argparse.Namespace, default: str | None) -> None:
    output = (args.output or ROOT / "artifacts" / args.config.stem).resolve()
    context = load_context(
        args.config,
        output,
        args.train_file,
        args.selection_file,
        args.device,
        args.split,
        args.reporting_file,
    )
    check_inputs(context.settings)
    roots = [
        discover(name)(context) for name in ([default] if default else args.contracts)
    ]
    runner = Orchestrator(
        roots,
        output / ".orchestrator",
        workers=args.workers,
        slots=args.slots,
        run_name=args.run_name,
        context=context,
    )
    runner.inspect()
    if args.plan:
        for key in runner.graph.order:
            state = runner.states[key]
            print(
                f"{state['state']:>8}  {runner.graph.nodes[key].label}: {state['reason']}"
            )
    else:
        try:
            runner.run(
                mode=_mode(args),
                allow_remote=args.allow_remote,
                interactive=sys.stdin.isatty(),
            )
        finally:
            if args.export_graph:
                runner.export(args.export_graph)
        _show_results(roots)
    if args.export_graph:
        runner.export(args.export_graph)


def main(default: str | None = None, arguments: list[str] | None = None) -> None:
    parser = _parser(default)
    args = parser.parse_args(arguments)
    if args.config is None:
        args.config = _choose_config(parser)
    try:
        _execute(args, default)
    except (
        DependencyError,
        ValueError,
        FileNotFoundError,
        ModuleNotFoundError,
    ) as error:
        parser.exit(1, f"{error}\n")


def entrypoint(filename: str) -> None:
    main(script_name(filename))
