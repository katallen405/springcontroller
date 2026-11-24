"""The Gen2 stability model reproduces what the arm did (plain Python, ~1 min):
with the measured ~60 ms loop delay, the gains used in every working 2026-11
test are stable and the code's old defaults are unstable -- and without the
delay the model would (wrongly) call both stable."""

import importlib.util
import sys
from pathlib import Path

import pytest

PATH = (Path(__file__).resolve().parents[1] / "springcontroller" / "tools"
        / "gen2_stability_model.py")
spec = importlib.util.spec_from_file_location("gen2_stability_model", PATH)
model = importlib.util.module_from_spec(spec)
sys.modules["gen2_stability_model"] = model
spec.loader.exec_module(model)

ONSET = model.POSES["onset"]


def test_working_gains_stable_with_measured_delay():
    swing, _ = model.simulate(ONSET, model.gains(60, 5), delay_ticks=6)
    assert swing < 1.0


def test_old_defaults_unstable_with_measured_delay():
    swing, _ = model.simulate(ONSET, model.gains(300, 30), delay_ticks=6)
    assert swing > 1.0


@pytest.mark.parametrize("k,b", [(60, 5), (300, 30)])
def test_without_delay_both_look_stable(k, b):
    # Documents the earlier modelling error: no delay, no instability.
    swing, _ = model.simulate(ONSET, model.gains(k, b), delay_ticks=0)
    assert swing < 1.0
