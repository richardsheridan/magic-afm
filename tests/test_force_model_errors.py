"""The force model raises ValueError on out-of-range parameters, asserts or not."""

# Copyright (C) Richard J. Sheridan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from magic_afm import calculation

REPO_ROOT = Path(__file__).resolve().parent.parent

# parameters of force_curve and delta_curve after the curve and its input, as the GUI
# Demo sets them
PARAMETER_NAMES = ("k", "radius", "tau", "M", "fc", "delta_shift", "force_shift", "lj")
VALID = dict(
    k=10.0,
    radius=20.0,
    tau=0.0,
    M=0.1,
    fc=2.0,
    delta_shift=0.0,
    force_shift=0.0,
    lj=1.0,
)
# parameter, out-of-range value, pattern the message must match
BAD_PARAMETERS = {
    "k-zero": ("k", 0.0, r"\bk\b"),
    "radius-zero": ("radius", 0.0, r"\bradius\b"),
    "M-zero": ("M", 0.0, r"\bM\b"),
    "fc-zero": ("fc", 0.0, r"\bfc\b"),
    "tau-negative": ("tau", -0.1, r"\btau\b"),
    "tau-above-one": ("tau", 1.1, r"\btau\b"),
}
MODEL_FUNCTIONS = ("force_curve", "delta_curve")
CASES = [(f, p) for f in MODEL_FUNCTIONS for p in BAD_PARAMETERS]


def call_model(function: str, **parameters):
    """Call calculation.force_curve or delta_curve as bound now, with parameters."""
    args = [{**VALID, **parameters}[name] for name in PARAMETER_NAMES]
    if function == "force_curve":
        delta = -15 * (np.cos(np.linspace(0, np.pi, 50)) + 0.5)
        return calculation.force_curve(calculation.red_extend, delta, *args)
    return calculation.delta_curve(calculation.schwarz_wrap, 1.0, *args)


def outcome(function: str, **parameters) -> dict:
    """The exception class names (its MRO) and message of call_model, if any."""
    try:
        result = call_model(function, **parameters)
    except Exception as exc:
        return {"mro": [cls.__name__ for cls in type(exc).__mro__], "message": str(exc)}
    return {"mro": None, "finite": bool(np.all(np.isfinite(result)))}


def all_outcomes() -> dict:
    """outcome of a valid call and of every case, keyed by function then case."""
    return {
        function: {
            "valid": outcome(function),
            **{
                case: outcome(function, **{name: value})
                for case, (name, value, _) in BAD_PARAMETERS.items()
            },
        }
        for function in MODEL_FUNCTIONS
    }


# Runs in a fresh interpreter under -O, so the jitting cannot reach the other tests:
# argv[1] is this directory. The last line of stdout is the JSON of every outcome, of
# the plain model and then, where numba is installed, of the jitted one.
OPTIMIZED_MODEL_SCRIPT = """
import importlib.util
import json
import sys
import types

sys.path.insert(0, sys.argv[1])
import test_force_model_errors as t

from magic_afm import calculation

out = {"optimize": sys.flags.optimize, "plain": t.all_outcomes(), "jitted": None}
if importlib.util.find_spec("numba") is not None:
    calculation.warmup_jit_worker()
    out["jitted_model"] = not isinstance(calculation.force_curve, types.FunctionType)
    out["jitted"] = t.all_outcomes()
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def optimized_outcomes():
    """Every outcome of the plain and jitted model, made under python -O."""
    proc = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            OPTIMIZED_MODEL_SCRIPT,
            str(Path(__file__).parent),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["optimize"] >= 1
    return out


@pytest.mark.parametrize(("function", "case"), CASES)
def test_force_model_rejects_out_of_range_parameter(function, case):
    name, value, match = BAD_PARAMETERS[case]
    with pytest.raises(ValueError, match=match):
        call_model(function, **{name: value})


@pytest.mark.parametrize(("function", "case"), CASES)
def test_force_model_rejects_out_of_range_parameter_without_asserts(
    optimized_outcomes, function, case
):
    assert optimized_outcomes["plain"][function]["valid"] == {
        "mro": None,
        "finite": True,
    }
    actual = optimized_outcomes["plain"][function][case]
    assert actual["mro"] is not None, "no exception under python -O"
    assert "ValueError" in actual["mro"], actual
    assert re.search(BAD_PARAMETERS[case][2], actual["message"]), actual


@pytest.mark.parametrize(("function", "case"), CASES)
def test_jitted_force_model_rejects_out_of_range_parameter_without_asserts(
    optimized_outcomes, function, case
):
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed, so warmup_jit_worker() jits nothing")
    assert optimized_outcomes["jitted_model"]
    assert optimized_outcomes["jitted"][function]["valid"] == {
        "mro": None,
        "finite": True,
    }
    actual = optimized_outcomes["jitted"][function][case]
    assert actual["mro"] is not None, "no exception from the jitted model"
    assert actual["mro"][0] == "ValueError", actual
