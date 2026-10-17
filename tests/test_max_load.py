"""The max-load quantities, pull-off delta and true surface z of a fitted curve."""

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

# max_load_quantities, pull_off_delta and true_surface_z replace calc_def_ind_ztru_ac
# and must give bitwise the values it gave. Its values differ between the plain and
# jitted model (about 1e-7 in indentation) and across platforms, so they are pinned by
# old_calc_def_ind_ztru_ac below, a copy of it as it stood before the split, run in the
# same process on the same inputs. It looks the model functions up on the module at
# call time, so it follows _warmup_jit_worker(). Before the split it was bitwise equal
# to calc_def_ind_ztru_ac on every case here, in env test plain and jitted and in env
# test-nonumba.

import importlib.util
import json
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_goldens as t2
from conftest import REPO_ROOT

from magic_afm import calculation
from magic_afm.calculation import FitMode


def old_calc_def_ind_ztru_ac(d, params, k, fit_mode, **kwargs):
    """calc_def_ind_ztru_ac as it stood before the split."""

    assert fit_mode
    force = d * k
    n_pts_max = len(force) // 25
    if fit_mode == FitMode.EXTEND:
        maxforce = force[-n_pts_max:].mean()
    elif fit_mode == FitMode.RETRACT:
        maxforce = force[:n_pts_max].mean()
    elif fit_mode == FitMode.BOTH:
        split = kwargs["split"]
        maxforce = force[split - n_pts_max // 2 : split + n_pts_max // 2].mean()
    else:
        raise ValueError("Unknown fit_mode: ", fit_mode)

    maxdelta = calculation.delta_curve(
        calculation._schwarz_wrap,
        maxforce,
        k,
        params["radius"],
        params["tau"],
        params["M"],
        params["fc"],
        params["delta_shift"],
        params["force_shift"],
        params["lj_scale"],
    )
    mindelta = calculation.delta_curve(
        calculation._schwarz_wrap,
        params["force_shift"] - params["fc"],
        k,
        params["radius"],
        params["tau"],
        params["M"],
        params["fc"],
        params["delta_shift"],
        params["force_shift"],
        params["lj_scale"],
    )
    # Identical on extend or retract, but in the case of `FitMode.BOTH` need to pick one
    zeroindforce = calculation.force_curve(
        calculation.red_retract,
        params["delta_shift"].squeeze(),
        k,
        params["radius"].squeeze(),
        params["tau"].squeeze(),
        params["M"].squeeze(),
        params["fc"].squeeze(),
        params["delta_shift"].squeeze(),
        params["force_shift"].squeeze(),
        params["lj_scale"].squeeze(),
    ).item()

    maxforce -= params["force_shift"]
    red_fc = (params["tau"] - 4) / 2
    ref_force = -params["fc"] / red_fc
    df = abs(maxforce / ref_force - red_fc)
    red_contact_radius = ((3 * red_fc + 6) ** (1 / 2) + df ** (1 / 2)) ** (2 / 3)
    contact_radius = red_contact_radius * (
        params["M"] / ref_force / params["radius"]
    ) ** (-1 / 3)

    deflection = (maxforce + params["fc"]) / k
    indentation = maxdelta - mindelta
    z_true_surface = params["delta_shift"] + zeroindforce / k
    return tuple(
        x.item()
        for x in (deflection, indentation, z_true_surface, mindelta, contact_radius)
    )


# Synthetic cases from test_goldens, fitted as there, and two with tau set after the
# fit: at the default tau = 0 the pull-off delta equals delta_shift, at tau > 0 not.
CASES = {
    "demo_r0_c0": ("demo_r0_c0", None),
    "stiff": ("stiff", None),
    "soft": ("soft", None),
    "dmt": ("dmt", None),
    "dmt_tau0.5": ("dmt", 0.5),
    "demo_r0_c0_tau1": ("demo_r0_c0", 1.0),
}


@cache
def fitted_inputs(case, mode):
    """(d, parms, split) of a fitted synthetic curve, parms from fit_force_curve."""
    golden_case, tau = CASES[case]
    curve = t2.demo_curve(**t2.SYNTHETIC_CASES[golden_case])
    z, d, split, _ = t2.preprocess(((0, 0), curve), mode)
    with np.errstate(all="ignore"):
        parms, *_ = calculation.fit_force_curve(
            z, d, k=t2.DEMO_K, fit_mode=mode, split=split, **t2.FIT_OPTIONS
        )
    if tau is not None:
        parms = parms.copy()
        parms["tau"] = tau
    return d, parms, split


def as_floats(values):
    return [np.asarray(x).item() for x in values]


def old_values(case, mode):
    """(deflection, indentation, contact_radius, pull-off delta, true surface z)."""
    d, parms, split = fitted_inputs(case, mode)
    with np.errstate(all="ignore"):
        defl, ind, ztru, mindelta, a_c = old_calc_def_ind_ztru_ac(
            d, parms, t2.DEMO_K, mode, split=split
        )
    return [defl, ind, a_c, mindelta, ztru]


def new_values(case, mode):
    """The same five, from max_load_quantities, pull_off_delta and true_surface_z."""
    d, parms, split = fitted_inputs(case, mode)
    with np.errstate(all="ignore"):
        max_load = calculation.max_load_quantities(
            d, parms, t2.DEMO_K, mode, split=split
        )
        mindelta = calculation.pull_off_delta(parms, t2.DEMO_K)
        ztru = calculation.true_surface_z(parms, t2.DEMO_K)
    return as_floats([*max_load, mindelta, ztru])


def assert_bitwise(actual, expected, where):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, err_msg=where)


def test_tau_cases_move_pull_off_delta_off_delta_shift():
    """The tau cases are the ones where the pull-off delta is not delta_shift."""
    for case, (_, tau) in CASES.items():
        for mode in t2.MODES:
            _, parms, _ = fitted_inputs(case, mode)
            mindelta = old_values(case, mode)[3]
            assert (mindelta == parms["delta_shift"].item()) == (tau is None)


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("case", CASES)
def test_max_load_quantities_match_calc_def_ind_ztru_ac_bitwise(case, mode):
    d, parms, split = fitted_inputs(case, mode)
    with np.errstate(all="ignore"):
        actual = calculation.max_load_quantities(d, parms, t2.DEMO_K, mode, split=split)
    assert len(actual) == 3
    assert all(isinstance(x, float) for x in actual)
    assert_bitwise(as_floats(actual), old_values(case, mode)[:3], f"{case} {mode.name}")


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("case", CASES)
def test_pull_off_delta_matches_calc_def_ind_ztru_ac_bitwise(case, mode):
    _, parms, _ = fitted_inputs(case, mode)
    with np.errstate(all="ignore"):
        actual = calculation.pull_off_delta(parms, t2.DEMO_K)
    assert isinstance(actual, float)
    assert_bitwise(actual, old_values(case, mode)[3], case)


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("case", CASES)
def test_true_surface_z_matches_calc_def_ind_ztru_ac_bitwise(case, mode):
    _, parms, _ = fitted_inputs(case, mode)
    with np.errstate(all="ignore"):
        actual = calculation.true_surface_z(parms, t2.DEMO_K)
    assert isinstance(actual, float)
    assert_bitwise(actual, old_values(case, mode)[4], case)


# Runs in a fresh interpreter, so the rebinding cannot reach the other tests: argv[1] is
# this directory. The last line of stdout is the JSON of the old and new values.
JITTED_VALUES_SCRIPT = """
import json
import sys
import types

sys.path.insert(0, sys.argv[1])
import test_max_load as tm

from magic_afm import calculation

calculation._warmup_jit_worker()
values = {
    case: {
        mode.name: {"old": tm.old_values(case, mode), "new": tm.new_values(case, mode)}
        for mode in tm.t2.MODES
    }
    for case in tm.CASES
}
jitted = not isinstance(calculation.delta_curve, types.FunctionType)
print(json.dumps({"jitted": jitted, "values": values}))
"""


@pytest.fixture(scope="module")
def jitted_values():
    """Old and new values of every case, after _warmup_jit_worker() in a subprocess."""
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed, so _warmup_jit_worker() jits nothing")
    proc = subprocess.run(
        [sys.executable, "-c", JITTED_VALUES_SCRIPT, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["jitted"]
    return result["values"]


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("case", CASES)
def test_jitted_split_functions_match_calc_def_ind_ztru_ac_bitwise(
    jitted_values, case, mode
):
    values = jitted_values[case][mode.name]
    assert_bitwise(values["new"], values["old"], f"jitted {case} {mode.name}")


def test_calc_def_ind_ztru_ac_is_gone():
    assert not hasattr(calculation, "calc_def_ind_ztru_ac")


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("case", ["demo_r0_c0", "dmt"])
def test_map_worker_properties_come_from_the_split_functions(case, mode):
    """The map worker exports the new functions' values, and TrueHeight negated."""
    curve = t2.demo_curve(**t2.SYNTHETIC_CASES[case])
    z, d, split, rc = t2.preprocess(((0, 0), curve), mode)
    options = t2.DEFAULT_OPTIONS | dict(k_sens=False)
    with np.errstate(all="ignore"):
        _, properties, parms, _ = calculation.fit_curve_properties(
            (z, d, split, rc), fit_mode=mode, k=t2.DEMO_K, **options
        )
        defl, ind, a_c = calculation.max_load_quantities(
            d, parms, t2.DEMO_K, mode, split=split
        )
        ztru = calculation.true_surface_z(parms, t2.DEMO_K)
    expected = dict(
        Deflection=defl,
        Indentation=ind,
        ContactRadius=a_c,
        TrueHeight=-np.asarray(ztru),
        IndentationRatio=defl / ind,
    )
    for name, value in expected.items():
        np.testing.assert_allclose(
            properties[name],
            np.float32(np.asarray(value).item()),
            rtol=0,
            atol=0,
            err_msg=f"{case} {mode.name} {name}",
        )
