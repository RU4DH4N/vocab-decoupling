import pytest

from framework.sources import ROOT, closure

GENERATION = {"execution/generation.py", "execution/streams.py", "execution/mauve.py"}
TRAINING = (
    "execution.training",
    "execution.baseline_training",
    "execution.planning_training",
    "execution.fitted_controls",
    "execution.bootstrap",
    "execution.clocks",
    "execution.interfaces",
    "execution.preparation",
)


def files(module):
    return {path.relative_to(ROOT).as_posix() for path in closure(module)}


@pytest.mark.parametrize("module", TRAINING)
def test_training_code_never_depends_on_generation(module):
    assert not files(module) & GENERATION


def test_measurement_code_includes_what_it_runs():
    measured = files("execution.receiver_experiment")
    assert {"execution/receiver_experiment.py", *GENERATION} <= measured
    assert "execution/training.py" in measured
    assert "models/protocol/receiver.py" in measured


def test_unknown_modules_are_rejected():
    with pytest.raises(ValueError, match="not a module"):
        closure("execution.missing")
