"""Golden baselines of fitted values, so that a numerics change shows as drift."""

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

# The values live in tests/goldens/t2_goldens.json (synthetic) and
# tests/goldens/<pixi platform>/t2_goldens.json (the .nid), written by
# tests/goldens/make_t2_goldens.py, which imports the builders below so the
# two cannot diverge. Regenerate on purpose only, and say why in the commit.
#
# Values are stored at the calculation layer, in fit units (nm, nN, GPa), as the CLI
# holds them in its property and parameter maps before export: no export units factor
# and no row flip. What the CLI writes to disk is pinned in tests/test_cli.py. See
# tests/goldens/README.md for the golden sets.
#
# The model runs unjitted here, as in CLI workers at the default (no --jit).
# warmup_jit_worker() rebinds module globals for the rest of the session, so
# it is never called under pytest; the generator measures that state apart.

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import load_platform_goldens

from magic_afm import calculation
from magic_afm.calculation import PROPERTY_DTYPE, FitFix, FitMode
from magic_afm.data_readers import SUFFIX_FVFILE_MAP

GOLDENS_PATH = Path(__file__).resolve().parent / "goldens" / "t2_goldens.json"
NID_RELPATH = "test_cases/20251216_141819_PVDF2_Film_U_00002.nid"
MODES = (FitMode.EXTEND, FitMode.RETRACT, FitMode.BOTH)

# Fit options at the CLI defaults (magic_afm/cli/_impl.py, main). The GUI
# defaults are the same, except M, which is unused while RADIUS is fixed.
DEFAULT_OPTIONS = dict(
    fit_fix=FitFix.DEFAULTS,
    radius=20.0,
    M=1.0,
    tau=0.0,
    lj_scale=2.0,
    vd=0.0,
    li_per=0.0,
    li_amp=0.0,
    drag=0.0,
    k_sens=True,
)

# The maps the CLI exports at DEFAULT_OPTIONS: every property, and the
# parameters (and their errors) that FitFix.DEFAULTS leaves free.
FIELDS = {
    "properties": PROPERTY_DTYPE.names,
    "parms": ("M", "fc", "delta_shift", "force_shift", "lj_scale"),
    "parms_err": ("M", "fc", "delta_shift", "force_shift", "lj_scale"),
}

# Demo curves as DemoStub and DemoForceVolumeFile build them in
# magic_afm/gui/_impl.py, with the force_curve parameters made variable.
DEMO_K = 10.0
DEMO_DELTA = -15 * (np.cos(np.linspace(0, np.pi * 2, 1000, endpoint=False)) + 0.5)
DEMO_PARMS = dict(radius=20, tau=0, M=0.1, fc=2, delta_shift=0, force_shift=0, lj=1)

# The demo_* cases are the GUI Demo itself at three of its pixels. The rest
# move the model: stiff gives an indentation ratio above 1 (the others are
# below), soft a ratio near 0.02, and dmt is generated at tau = 1 but fitted
# with tau fixed at the default 0.
SYNTHETIC_CASES = {
    "demo_r0_c0": dict(seed=(0, 0)),
    "demo_r5_c9": dict(seed=(5, 9)),
    "demo_r63_c63": dict(seed=(63, 63)),
    "stiff": dict(seed=(1, 1), M=1.0, fc=0.5),
    "soft": dict(seed=(3, 3), M=0.02, fc=5.0),
    "dmt": dict(seed=(2, 2), tau=1.0),
}
FINGERPRINT_STRIDE = {"raw": 100, "resampled": 32}

# Tolerances per stored field, (rtol, atol), in fit units.
#
# Measured spread, env test (model unjitted) against env test-nonumba, osx-arm64, on
# every value in the goldens files: bitwise equal, NaN patterns included, except
#   Indentation       nid: max rel 2.44e-5 (abs 2.4e-4 nm); synthetic: 0
#   IndentationRatio  nid: max rel 2.45e-5 (abs 1.3e-4); synthetic: 0
#   parms.delta_shift synthetic soft BOTH: rel 9.9e-8 (abs 3e-10 nm, 1 ulp)
# The nid fitted parameters are bitwise equal, so the Indentation spread arises after
# the fit, in calc_def_ind_ztru_ac (delta_curve over schwarz_wrap, jitted only with
# numba), on float32 parameters. Those two fields get rtol 1e-4, about 4x the spread.
# (Casting the nid curves to float64 does not shrink it: the fits then differ by up to
# 1% in M between the envs, so their agreement here depends on the float32 input path.)
#
# The atols cover fields whose value can sit near 0: about 8 float32 ulps at
# the field's natural scale (shifts and fc at the 10-20 nm or nN scale of the
# curves, lj_scale on [-6, 6]). SensIndMod_k is (M' - M) / M / 1e-3 from
# float32 M, so one ulp of M moves it by about 1.2e-4 at any magnitude; it
# gets atol 1e-3.
#
# Across operating systems the spread is far larger than numba against plain. The
# synthetic values are shared by every platform, so these tolerances span OSes: goldens
# written in CI by every Linux and Windows x86-64 env, against these osx-arm64 values,
# keep every NaN pattern, but the resampled curves already differ by up to rel 3.2e-6
# and the fits amplify that. At the atols here, the rtol each field needs is at most
# 2.2e-5 (parms_err.lj_scale) and 9e-5 for SensIndMod_k, so every field gets rtol 1e-4
# (4.5x) and SensIndMod_k 1e-3. Within one OS, numba against plain and 3.11 against 3.14
# agree to 4e-7. One case is ill-conditioned and gets its own tolerances
# (CASE_TOLERANCES). Tightening these is for a deliberate re-baseline after the planned
# numerics changes (resampling, the numba and plain paths).
#
# For the jitted model: with warmup_jit_worker() (GUI workers, CLI --jit) the spread
# against the unjitted numba state is at most rel 3.2e-7 in Indentation and
# IndentationRatio and nothing in the fitted parameters, so that state passes too.
SHIFT_ATOL = 1e-5
CROSS_OS_RTOL = 1e-4
TOLERANCES: dict[tuple[str, str], tuple[float, float]] = {
    ("properties", "Deflection"): (CROSS_OS_RTOL, 0.0),
    ("properties", "Indentation"): (CROSS_OS_RTOL, 0.0),
    ("properties", "ContactRadius"): (CROSS_OS_RTOL, 0.0),
    ("properties", "TrueHeight"): (CROSS_OS_RTOL, SHIFT_ATOL),
    ("properties", "IndentationRatio"): (CROSS_OS_RTOL, 0.0),
    ("properties", "SensIndMod_k"): (1e-3, 1e-3),
    ("properties", "SumSquaredError"): (CROSS_OS_RTOL, 0.0),
    ("parms", "M"): (CROSS_OS_RTOL, 0.0),
    ("parms", "fc"): (CROSS_OS_RTOL, SHIFT_ATOL),
    ("parms", "delta_shift"): (CROSS_OS_RTOL, SHIFT_ATOL),
    ("parms", "force_shift"): (CROSS_OS_RTOL, SHIFT_ATOL),
    ("parms", "lj_scale"): (CROSS_OS_RTOL, 4e-6),
    ("parms_err", "M"): (CROSS_OS_RTOL, 0.0),
    ("parms_err", "fc"): (CROSS_OS_RTOL, 0.0),
    ("parms_err", "delta_shift"): (CROSS_OS_RTOL, 0.0),
    ("parms_err", "force_shift"): (CROSS_OS_RTOL, 0.0),
    ("parms_err", "lj_scale"): (CROSS_OS_RTOL, 0.0),
}
# The stiff curve fitted in BOTH is ill-conditioned: across OSes its fitted M and the
# properties that follow from it stay within the tolerances above (at most rel 1.9e-5,
# in Indentation), but TrueHeight, SensIndMod_k, the other fitted parameters and all
# the errors need far more. At the atols above, the rtol each of those needs is
#   properties  TrueHeight 2.4e-3, SensIndMod_k 2.1e-2
#   parms       fc 1.2e-3, delta_shift 4.4e-3, force_shift 1.6e-3, lj_scale 2.8e-3
#   parms_err   M 1.9e-3, fc 4.9e-3, delta_shift 3.4e-3, force_shift 4.2e-3,
#               lj_scale 2.7e-2
# with Linux farthest from these osx-arm64 values (alike in all three envs; Windows
# about 0.8x of that). Each gets rtol about 4x its need, rounded to 1, 2 or 5 in the
# leading digit, and keeps its atol.
STIFF_BOTH_RTOL = {
    ("properties", "TrueHeight"): 1e-2,
    ("properties", "SensIndMod_k"): 1e-1,
    ("parms", "fc"): 5e-3,
    ("parms", "delta_shift"): 2e-2,
    ("parms", "force_shift"): 1e-2,
    ("parms", "lj_scale"): 1e-2,
    ("parms_err", "M"): 1e-2,
    ("parms_err", "fc"): 2e-2,
    ("parms_err", "delta_shift"): 2e-2,
    ("parms_err", "force_shift"): 2e-2,
    ("parms_err", "lj_scale"): 1e-1,
}
CASE_TOLERANCES: dict[tuple[str, str], dict[tuple[str, str], tuple[float, float]]] = {
    ("stiff", "BOTH"): TOLERANCES
    | {key: (rtol, TOLERANCES[key][1]) for key, rtol in STIFF_BOTH_RTOL.items()},
}
# Curve samples: spread 0 in both. raw is float64 from numpy's RNG stream and
# the model; resampled is soxr output (float64 for Demo, float32 for the nid).
FINGERPRINT_TOLERANCES = {"raw": (1e-12, 1e-12), "resampled": (1e-6, 1e-6)}


def demo_curve(seed, **parms):
    """One Demo curve, (zext, zret), (dext, dret), as the GUI Demo makes it."""
    p = {**DEMO_PARMS, **parms}
    gen = np.random.default_rng(seed=seed)
    fc_parms = (DEMO_K, p["radius"], p["tau"], p["M"], p["fc"])
    fc_parms += (p["delta_shift"], p["force_shift"], p["lj"])
    deltaext = DEMO_DELTA[: DEMO_DELTA.size // 2]
    deltaret = DEMO_DELTA[DEMO_DELTA.size // 2 :]
    fext = calculation.force_curve(calculation.red_extend, deltaext, *fc_parms)
    fret = calculation.force_curve(calculation.red_retract, deltaret, *fc_parms)
    dext = fext / DEMO_K + gen.normal(scale=0.1, size=fext.size)
    dret = fret / DEMO_K + gen.normal(scale=0.1, size=fret.size)
    zext = deltaext + dext + gen.normal(scale=0.01, size=fext.size)
    zret = deltaret + dret + gen.normal(scale=0.01, size=fret.size)
    return (zext, zret), (dext, dret)


def preprocess(rc_curve, fit_mode):
    """Resample and select a segment as the CLI does, with s_ratio 1."""
    return calculation.process_force_curve(rc_curve, fit_mode=fit_mode, s_ratio=1.0)


def fit_one(z_d_s_rc, fit_mode, k):
    """Fit one preprocessed curve as a CLI worker does; NaN fields on failure."""
    with np.errstate(all="ignore"):  # cli_init sets this in every worker
        _, properties, parms, parms_err = calculation.calc_properties_imap(
            z_d_s_rc, fit_mode=fit_mode, k=k, **DEFAULT_OPTIONS
        )
    results = {"properties": properties, "parms": parms, "parms_err": parms_err}
    return {
        group: {
            name: np.nan if results[group] is None else results[group][name].item()
            for name in names
        }
        for group, names in FIELDS.items()
    }


def fit_synthetic(case, fit_mode):
    curve = demo_curve(**SYNTHETIC_CASES[case])
    return fit_one(preprocess(((0, 0), curve), fit_mode), fit_mode, DEMO_K)


def raw_fingerprint(case):
    (zext, zret), (dext, dret) = demo_curve(**SYNTHETIC_CASES[case])
    arrays = dict(zext=zext, zret=zret, dext=dext, dret=dret)
    return {name: a[:: FINGERPRINT_STRIDE["raw"]] for name, a in arrays.items()}


def resampled_fingerprint(rc_curve, fit_mode):
    z, d, split, _ = preprocess(rc_curve, fit_mode)
    stride = FINGERPRINT_STRIDE["resampled"]
    return dict(z=z[::stride], d=d[::stride], split=split)


def open_fvfile(path):
    fvfile_cls, opener = SUFFIX_FVFILE_MAP[path.suffix.lower()]
    return fvfile_cls, opener(path)


def nid_results(path, fit_mode):
    """Resampled fingerprint of pixel (0, 0) and every exported map, as arrays."""
    fvfile_cls, data = open_fvfile(path)
    with data:
        fvfile = fvfile_cls.parse(data)
        volume = fvfile.volumes[0]  # the CLI's default trace selection
        maps = {
            group: {name: np.full(volume.shape, np.nan) for name in names}
            for group, names in FIELDS.items()
        }
        fingerprint = None
        for rc_curve in volume.iter_curves():
            rc = rc_curve[0]
            if rc == (0, 0):
                fingerprint = resampled_fingerprint(rc_curve, fit_mode)
            fitted = fit_one(preprocess(rc_curve, fit_mode), fit_mode, fvfile.k)
            for group, values in fitted.items():
                for name, value in values.items():
                    maps[group][name][rc] = value
    return fingerprint, maps


def load_goldens():
    with GOLDENS_PATH.open() as f:
        return json.load(f)


def as_array(values, dtype):
    """JSON lists back to arrays; null is NaN."""
    return np.array(values, dtype=dtype)


@pytest.fixture(scope="module")
def goldens():
    return load_goldens()


@pytest.fixture(autouse=True)
def default_resample_npts():
    assert calculation.RESAMPLE_NPTS == 512, (
        "goldens are captured at 512 resampled points; unset MAFM_RESAMPLE_NPTS"
    )


def assert_fields_match(actual, golden, where, tolerances=TOLERANCES):
    for group, names in FIELDS.items():
        for name in names:
            got = np.asarray(actual[group][name], dtype=np.float32)
            want = as_array(golden[group][name], np.float32)
            label = f"{where} {group}.{name}"
            np.testing.assert_array_equal(
                np.isnan(got), np.isnan(want), err_msg=f"{label}: NaN pattern"
            )
            rtol, atol = tolerances[group, name]
            np.testing.assert_allclose(
                got, want, rtol=rtol, atol=atol, equal_nan=True, err_msg=label
            )


def assert_fingerprint_matches(actual, golden, kind, where):
    rtol, atol = FINGERPRINT_TOLERANCES[kind]
    for name, want in golden.items():
        if name == "split":
            assert actual[name] == want, where
            continue
        np.testing.assert_allclose(
            actual[name],
            as_array(want, np.float64),
            rtol=rtol,
            atol=atol,
            err_msg=f"{where} {name}",
        )


@pytest.mark.parametrize("case", SYNTHETIC_CASES)
def test_demo_curve_samples_match_golden(goldens, case):
    golden = goldens["synthetic"][case]["raw"]
    assert_fingerprint_matches(raw_fingerprint(case), golden, "raw", case)


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_resampled_demo_curve_samples_match_golden(goldens, mode):
    case = "demo_r0_c0"
    golden = goldens["synthetic"][case]["resampled"][mode.name]
    curve = demo_curve(**SYNTHETIC_CASES[case])
    actual = resampled_fingerprint(((0, 0), curve), mode)
    assert_fingerprint_matches(actual, golden, "resampled", f"{case} {mode.name}")


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("case", SYNTHETIC_CASES)
def test_synthetic_fit_matches_golden(goldens, case, mode):
    golden = goldens["synthetic"][case]["fits"][mode.name]
    tolerances = CASE_TOLERANCES.get((case, mode.name), TOLERANCES)
    actual = fit_synthetic(case, mode)
    assert_fields_match(actual, golden, f"{case} {mode.name}", tolerances)


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_fitfun_raises_no_deprecation_warning(mode):
    """One fit, with errors raised rather than turned into NaN, does not warn."""
    curve = demo_curve(**SYNTHETIC_CASES["demo_r0_c0"])
    z, d, split, _ = preprocess(((0, 0), curve), mode)
    with np.errstate(all="ignore"):
        parms, parms_err, sse, _ = calculation.fitfun(
            z, d, k=DEMO_K, fit_mode=mode, split=split, **DEFAULT_OPTIONS
        )
    assert np.all(np.isfinite(parms.item()))
    assert np.all(np.isfinite(parms_err.item()))
    assert np.isfinite(sse)


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_synthetic_fit_with_deprecations_as_errors_matches_golden(goldens, mode):
    """The CLI fit path gives finite golden values when deprecations are errors."""
    case = "demo_r0_c0"
    golden = goldens["synthetic"][case]["fits"][mode.name]
    actual = fit_synthetic(case, mode)
    nonfinite = [
        f"{group}.{name}"
        for group, values in actual.items()
        for name, value in values.items()
        if not np.isfinite(value)
    ]
    assert not nonfinite, f"fit fell back to NaN: {nonfinite}"
    assert_fields_match(actual, golden, f"{case} {mode.name} deprecations as errors")


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_nid_property_maps_match_golden(real_data_file, mode):
    golden = load_platform_goldens("t2_goldens.json")["nid"][mode.name]
    path = Path(real_data_file(NID_RELPATH))
    fingerprint, maps = nid_results(path, mode)
    where = f"nid {mode.name}"
    assert_fingerprint_matches(fingerprint, golden["resampled"], "resampled", where)
    assert_fields_match(maps, golden["maps"], where)
