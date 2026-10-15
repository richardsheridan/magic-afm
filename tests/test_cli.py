"""Characterisation of what the CLI writes today, captured from the code at T3."""

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

# These tests pin the CLI's observable output (file names, array shapes and
# dtypes, units, the options JSON, and values) so that B3 can move the export
# code without changing it. They are not a statement that today's output is
# right: see the notes on units, trace and error handling below.
#
# Values live in tests/goldens/t3_goldens.json, written by
# tests/goldens/make_t3_goldens.py, which imports the runs and summaries
# below. Regenerate on purpose only, and say why in the commit.
#
# One complete file per reader path that works through the CLI today: ARDF (both0001:
# two volumes, so --trace matters), ARH5 (rtrace0001.h5), Nanoscope QNM
# (onekhz.000.pfc), Nanoscope force volume (A1...spm) and NID (the example .nid, also
# the subject of T2). Excluded, per the plan: DOE4 (mis-decoded), the stopped
# hard/mellow scans (hard0001 and hard0003 are covered by test_stopped_scans.py since
# D4) and the non-square twoxln/twoxpt (the CLI crashes until B2a).
#
# Every in-process run passes --disable-progress (tqdm only) and an absolute
# --output-path under pytest's tmp dir; nothing else is restricted. The one
# subprocess run keeps the progress bars.

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import imageio
import numpy as np
import pytest
from click.testing import CliRunner, Result

from magic_afm.calculation import FitFix

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDENS_PATH = REPO_ROOT / "tests" / "goldens" / "t3_goldens.json"
T2_GOLDENS_PATH = REPO_ROOT / "tests" / "goldens" / "t2_goldens.json"

ARDF_RELPATH = "test_cases/PDMS_SAA/ffm/both0001.ARDF"
ARH5_RELPATH = "test_cases/PDMS_SAA/ffm/rtrace0001.h5"
QNM_RELPATH = "test_cases/PS_PCL_SAA/QNM/onekhz.000.pfc"
FFV_RELPATH = "test_cases/A1_OCSOH-PMMA21k_300421.0_300425.spm"
NID_RELPATH = "PVDF2_Film_U_00002/20251216_141819_PVDF2_Film_U_00002.nid"
NID_STEM = "20251216_141819_PVDF2_Film_U_00002"

# name: (relpath, extra CLI arguments)
CLI_RUNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ardf": (ARDF_RELPATH, ()),
    "ardf_trace": (ARDF_RELPATH, ("--trace", "trace")),
    "ardf_retrace": (ARDF_RELPATH, ("--trace", "retrace")),
    "arh5": (ARH5_RELPATH, ()),
    "arh5_retrace": (ARH5_RELPATH, ("--trace", "retrace")),
    "qnm": (QNM_RELPATH, ()),
    "ffv": (FFV_RELPATH, ()),
    "nid": (NID_RELPATH, ()),
    "nid_extend": (NID_RELPATH, ("--fit-mode", "extend")),
    "nid_both": (NID_RELPATH, ("--fit-mode", "both")),
    # every fixable parameter freed except lj_scale, which is fixed
    "nid_flags": (
        NID_RELPATH,
        (
            "-fit-radius",
            "-fit-tau",
            "-fix-lj-scale",
            "-fit-vd",
            "-fit-li-per",
            "-fit-li-amp",
            "-fit-drag",
            "--no-k-sens",
        ),
    ),
    **{
        f"nid_{ext}": (NID_RELPATH, ("--output-type", ext))
        for ext in ("npz", "csv", "tsv", "txt", "asc", "tif")
    },
}

# The runs whose values are stored in the goldens file, as summaries for the
# 32 x 32 maps and in full for the 10 x 10 nid.
SUMMARY_RUNS = ("ardf", "ardf_retrace", "arh5", "qnm", "ffv")
FULL_RUNS = ("nid",)

# Pixels left out of every value comparison, in exported (row-flipped)
# coordinates. A1 pixel (12, 3) as the reader indexes it has an INT32_MIN
# extend half that the reader scales to -10,324 nm; B2f makes it NaN. The
# export flips rows, so it lands at (31 - 12, 3).
EXCLUDED_PIXELS = {"ffv": ((19, 3),)}

# Files written at the default options, in the order the CLI writes them:
# every property, then the fitted parameters FitFix.DEFAULTS leaves free,
# then their errors. M is exported as IndentationModulus, fc as
# AdhesionForce.
DEFAULT_MAPS = (
    "Deflection",
    "Indentation",
    "ContactRadius",
    "TrueHeight",
    "IndentationRatio",
    "SensIndMod_k",
    "SumSquaredError",
    "IndentationModulus",
    "AdhesionForce",
    "delta_shift",
    "force_shift",
    "lj_scale",
    "IndentationModulusErr",
    "AdhesionForceErr",
    "delta_shiftErr",
    "force_shiftErr",
    "lj_scaleErr",
)

# Units as written: fit units (nm, nN) for everything except the two maps
# with a factor, IndentationModulus in Pa (M is GPa in the fit) and
# AdhesionForce in N (fc is nN). So force_shift stays in nN although
# PARMS_UNITS_DICT says N, and lengths are nm although it says m (A1).
# exported map: (T2 goldens group, field, factor from fit units)
EXPORT_FIELDS = {
    "Deflection": ("properties", "Deflection", 1.0),
    "Indentation": ("properties", "Indentation", 1.0),
    "ContactRadius": ("properties", "ContactRadius", 1.0),
    "TrueHeight": ("properties", "TrueHeight", 1.0),
    "IndentationRatio": ("properties", "IndentationRatio", 1.0),
    "SensIndMod_k": ("properties", "SensIndMod_k", 1.0),
    "SumSquaredError": ("properties", "SumSquaredError", 1.0),
    "IndentationModulus": ("parms", "M", 1e9),
    "AdhesionForce": ("parms", "fc", 1e-9),
    "delta_shift": ("parms", "delta_shift", 1.0),
    "force_shift": ("parms", "force_shift", 1.0),
    "lj_scale": ("parms", "lj_scale", 1.0),
    "IndentationModulusErr": ("parms_err", "M", 1e9),
    "AdhesionForceErr": ("parms_err", "fc", 1e-9),
    "delta_shiftErr": ("parms_err", "delta_shift", 1.0),
    "force_shiftErr": ("parms_err", "force_shift", 1.0),
    "lj_scaleErr": ("parms_err", "lj_scale", 1.0),
}

# Tolerances per exported map, (rtol, atol) in exported units: T2's fit-unit
# tolerances (tests/test_goldens.py) with atol scaled by the export factor,
# except Indentation and IndentationRatio.
#
# Measured spread on every pixel of every run stored here, env test (numba
# 0.67.0) against env test-nonumba, osx-arm64: bitwise equal, NaN patterns
# included, except Indentation and IndentationRatio, as in T2. Their max rel
# is 5.4e-5 on onekhz.000.pfc, 2.6e-5 on A1...spm, 3.2e-5 on the nid and
# 1.5e-6 on both0001.ARDF and rtrace0001.h5. T2's rtol 1e-4 would leave less
# than 2x on the .pfc, so these two get 2e-4 here, about 3.7x the spread.
SHIFT_ATOL = 1e-5
FIT_TOLERANCES = {
    ("properties", "Deflection"): (1e-6, 0.0),
    ("properties", "Indentation"): (2e-4, 0.0),
    ("properties", "ContactRadius"): (1e-6, 0.0),
    ("properties", "TrueHeight"): (1e-6, SHIFT_ATOL),
    ("properties", "IndentationRatio"): (2e-4, 0.0),
    ("properties", "SensIndMod_k"): (1e-6, 1e-3),
    ("properties", "SumSquaredError"): (1e-6, 0.0),
    ("parms", "M"): (1e-6, 0.0),
    ("parms", "fc"): (1e-6, SHIFT_ATOL),
    ("parms", "delta_shift"): (1e-6, SHIFT_ATOL),
    ("parms", "force_shift"): (1e-6, SHIFT_ATOL),
    ("parms", "lj_scale"): (1e-6, 4e-6),
    ("parms_err", "M"): (1e-6, 0.0),
    ("parms_err", "fc"): (1e-6, 0.0),
    ("parms_err", "delta_shift"): (1e-6, 0.0),
    ("parms_err", "force_shift"): (1e-6, 0.0),
    ("parms_err", "lj_scale"): (1e-6, 0.0),
}


def map_tolerance(name):
    group, field, factor = EXPORT_FIELDS[name]
    rtol, atol = FIT_TOLERANCES[group, field]
    return rtol, atol * factor


QUANTILES = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
N_SAMPLES = 16


def export_name(stem, name, *, mode="Ret", trace="", ext="npy"):
    return f"{stem}_{mode}{name}{trace}.{ext}"


def options_name(stem, *, mode="Ret", trace=""):
    return f"{stem}_options{mode}{trace}.json"


class CliRun(NamedTuple):
    path: Path  # the input file
    out: Path  # the --output-path directory
    result: Result


def invoke_cli(path, out, args) -> Result:
    from magic_afm.cli._impl import main

    argv = ["--output-path", str(out), "--disable-progress", *args, str(path)]
    return CliRunner().invoke(main, argv, catch_exceptions=False)


def load_map(path):
    ext = path.suffix[1:]
    if ext == "npy":
        return np.load(path)
    if ext == "npz":
        with np.load(path) as npz:
            assert npz.files == ["arr_0"]
            return npz["arr_0"]
    if ext == "tif":
        return np.asarray(imageio.v2.imread(path))
    delimiter = {"csv": ",", "tsv": "\t"}.get(ext)
    return np.loadtxt(path, delimiter=delimiter, dtype=np.float32)


def excluded(run_name, a):
    """A float64 copy of a with the run's excluded pixels set to NaN."""
    a = np.array(a, dtype=np.float64)
    for rc in EXCLUDED_PIXELS.get(run_name, ()):
        a[rc] = np.nan
    return a


def nan_mask(a):
    """Rows of the NaN pattern, x for NaN and . for a value."""
    return ["".join("x" if v else "." for v in row) for row in np.isnan(a)]


def sample_pixels(mask_rows, n=N_SAMPLES):
    """n pixels spread evenly over the non-NaN ones, row-major."""
    finite = [
        (r, c)
        for r, row in enumerate(mask_rows)
        for c, ch in enumerate(row)
        if ch == "."
    ]
    idx = np.linspace(0, len(finite) - 1, min(n, len(finite))).round().astype(int)
    return [list(finite[i]) for i in idx]


def summarize(a, pixels):
    finite = a[~np.isnan(a)]
    return {
        "nan_mask": nan_mask(a),
        "quantiles": np.quantile(finite, QUANTILES).tolist(),
        "samples": [float(a[r, c]) for r, c in pixels],
    }


@pytest.fixture(scope="module")
def _cli_runs(tmp_path_factory):
    return tmp_path_factory, {}


@pytest.fixture
def run_cli(_cli_runs, real_data_file):
    """run_cli(name) runs CLI_RUNS[name] once per module and returns a CliRun."""
    tmp_path_factory, cache = _cli_runs

    def run(name) -> CliRun:
        if name not in cache:
            relpath, args = CLI_RUNS[name]
            path = Path(real_data_file(relpath))
            out = tmp_path_factory.mktemp(name) / "out"
            cache[name] = CliRun(path, out, invoke_cli(path, out, args))
        return cache[name]

    return run


@pytest.fixture(scope="module")
def goldens():
    with GOLDENS_PATH.open() as f:
        return json.load(f)


def written(run):
    return sorted(p.name for p in run.out.iterdir())


# run: (file stem, fit mode part, trace part, extension) for every run that
# fits and writes at the default options
NAMING = {
    "ardf": ("both0001", "Ret", "", "npy"),
    "ardf_trace": ("both0001", "Ret", "Trace", "npy"),
    "ardf_retrace": ("both0001", "Ret", "Retrace", "npy"),
    "arh5": ("rtrace0001", "Ret", "", "npy"),
    "qnm": ("onekhz.000", "Ret", "", "npy"),
    "ffv": ("A1_OCSOH-PMMA21k_300421.0_300425", "Ret", "", "npy"),
    "nid": (NID_STEM, "Ret", "", "npy"),
    "nid_extend": (NID_STEM, "Ext", "", "npy"),
    "nid_both": (NID_STEM, "Both", "", "npy"),
    **{
        f"nid_{ext}": (NID_STEM, "Ret", "", ext)
        for ext in ("npz", "csv", "tsv", "txt", "asc", "tif")
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMING)
def test_cli_writes_options_json_and_one_file_per_default_map(run_cli, name):
    stem, mode, trace, ext = NAMING[name]
    run = run_cli(name)
    assert run.result.exit_code == 0
    assert run.result.output == ""  # silent without --verbose
    expected = [
        export_name(stem, m, mode=mode, trace=trace, ext=ext) for m in DEFAULT_MAPS
    ]
    expected.append(options_name(stem, mode=mode, trace=trace))
    assert written(run) == sorted(expected)


@pytest.mark.slow
def test_cli_name_filter_follows_fit_fix_and_k_sens(run_cli):
    """Free parameters are exported with errors; M only while radius is fixed."""
    run = run_cli("nid_flags")
    stem = NID_STEM
    maps = [
        *("Deflection", "Indentation", "ContactRadius", "TrueHeight"),
        *("IndentationRatio", "SumSquaredError"),  # no SensIndMod_k
    ]
    parms = [
        *("radius", "tau", "AdhesionForce", "delta_shift", "force_shift"),
        *("vd", "li_per", "li_amp", "li_pha", "drag"),
    ]  # no IndentationModulus, no lj_scale
    maps += parms + [p + "Err" for p in parms]
    expected = [export_name(stem, m) for m in maps] + [options_name(stem)]
    assert run.result.exit_code == 0
    assert written(run) == sorted(expected)
    options = json.loads((run.out / options_name(stem)).read_text())
    assert options["fit_fix"] == FitFix.LJ_SCALE == 4
    assert options["k_sens"] is False


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n, v in NAMING.items() if v[3] == "npy"])
def test_cli_maps_are_float32_with_the_volume_shape(run_cli, name):
    stem, mode, trace, ext = NAMING[name]
    run = run_cli(name)
    shape = (10, 10) if stem == NID_STEM else (32, 32)
    for m in DEFAULT_MAPS:
        a = np.load(run.out / export_name(stem, m, mode=mode, trace=trace))
        assert (a.shape, a.dtype) == (shape, np.float32), m


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n, v in NAMING.items() if v[3] == "npy"])
def test_cli_options_json_matches_golden(run_cli, goldens, name):
    from magic_afm._options import OPTIONS_JSON_SCHEMA, load_options

    stem, mode, trace, _ = NAMING[name]
    run = run_cli(name)
    text = (run.out / options_name(stem, mode=mode, trace=trace)).read_text()
    options = json.loads(text)
    assert list(options) == list(OPTIONS_JSON_SCHEMA)
    assert options == goldens["options"][name]
    with (run.out / options_name(stem, mode=mode, trace=trace)).open() as f:
        assert load_options(f).keys() == OPTIONS_JSON_SCHEMA.keys()


@pytest.mark.slow
def test_cli_options_json_at_defaults_on_the_nid(run_cli):
    """The values the CLI records: per-file k and defl_sens, enums by name."""
    run = run_cli("nid")
    options = json.loads((run.out / options_name(NID_STEM)).read_text())
    assert options == {
        "k": 25.03,
        "defl_sens": 169.0,
        "sync_dist": None,
        "trace": None,
        "k_sens": True,
        "radius": 20.0,
        "M": 1.0,
        "tau": 0.0,
        "lj_scale": 2.0,
        "vd": 0.0,
        "li_per": 0.0,
        "li_amp": 0.0,
        "drag": 0.0,
        "fit_fix": FitFix.DEFAULTS.value,
        "fit_mode": "RETRACT",
    }


def run_maps(run, name):
    stem, mode, trace, ext = NAMING[name]
    return {
        m: excluded(
            name,
            load_map(run.out / export_name(stem, m, mode=mode, trace=trace, ext=ext)),
        )
        for m in DEFAULT_MAPS
    }


def assert_summary_matches(actual, golden, pixels, name, where):
    rtol, atol = map_tolerance(name)
    label = f"{where} {name}"
    assert nan_mask(actual) == golden["nan_mask"], f"{label}: NaN pattern"
    np.testing.assert_allclose(
        np.quantile(actual[~np.isnan(actual)], QUANTILES),
        golden["quantiles"],
        rtol=rtol,
        atol=atol,
        err_msg=f"{label} quantiles",
    )
    np.testing.assert_allclose(
        [actual[r, c] for r, c in pixels],
        golden["samples"],
        rtol=rtol,
        atol=atol,
        err_msg=f"{label} samples",
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", SUMMARY_RUNS)
def test_cli_map_values_match_golden_summary(run_cli, goldens, name):
    golden = goldens["summaries"][name]
    maps = run_maps(run_cli(name), name)
    for m in DEFAULT_MAPS:
        assert_summary_matches(maps[m], golden["maps"][m], golden["pixels"], m, name)


@pytest.mark.slow
def test_cli_nid_maps_match_golden_in_full(run_cli, goldens):
    golden = goldens["full"]["nid"]
    maps = run_maps(run_cli("nid"), "nid")
    for m in DEFAULT_MAPS:
        want = np.array(golden[m], dtype=np.float64)
        np.testing.assert_array_equal(np.isnan(maps[m]), np.isnan(want), err_msg=m)
        rtol, atol = map_tolerance(m)
        np.testing.assert_allclose(maps[m], want, rtol=rtol, atol=atol, err_msg=m)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "t2_mode"),
    [("nid_extend", "EXTEND"), ("nid", "RETRACT"), ("nid_both", "BOTH")],
)
def test_cli_nid_maps_are_t2_maps_scaled_and_row_flipped(run_cli, name, t2_mode):
    """Exports are the calculation-layer maps times a units factor, rows reversed."""
    with T2_GOLDENS_PATH.open() as f:
        t2_maps = json.load(f)["nid"][t2_mode]["maps"]
    maps = run_maps(run_cli(name), name)
    for m, (group, field, factor) in EXPORT_FIELDS.items():
        want = factor * np.array(t2_maps[group][field], dtype=np.float64)[::-1]
        rtol, atol = map_tolerance(m)
        np.testing.assert_array_equal(np.isnan(maps[m]), np.isnan(want), err_msg=m)
        np.testing.assert_allclose(maps[m], want, rtol=rtol, atol=atol, err_msg=m)


@pytest.mark.slow
def test_cli_trace_option_picks_the_volume_and_only_renames_for_trace(run_cli):
    """On a two-volume ARDF, --trace trace is the default volume, renamed."""
    stem = NAMING["ardf"][0]
    default, trace, retrace = (
        run_cli("ardf"),
        run_cli("ardf_trace"),
        run_cli("ardf_retrace"),
    )
    for m in DEFAULT_MAPS:
        a = np.load(default.out / export_name(stem, m))
        b = np.load(trace.out / export_name(stem, m, trace="Trace"))
        c = np.load(retrace.out / export_name(stem, m, trace="Retrace"))
        np.testing.assert_array_equal(a, b, err_msg=m)
        assert not np.array_equal(a, c, equal_nan=True), m


@pytest.mark.slow
@pytest.mark.parametrize("ext", ["npz", "tif"])
def test_cli_binary_output_types_hold_the_npy_values_exactly(run_cli, ext):
    stem = NID_STEM
    npy, other = run_cli("nid"), run_cli(f"nid_{ext}")
    for m in DEFAULT_MAPS:
        a = np.load(npy.out / export_name(stem, m))
        b = load_map(other.out / export_name(stem, m, ext=ext))
        assert b.dtype == np.float32, m
        np.testing.assert_array_equal(a, b, err_msg=m)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("ext", "sep"), [("csv", ","), ("tsv", "\t"), ("txt", " "), ("asc", " ")]
)
def test_cli_text_output_types_are_npy_values_at_8_significant_digits(
    run_cli, ext, sep
):
    stem = NID_STEM
    npy, other = run_cli("nid"), run_cli(f"nid_{ext}")
    for m in DEFAULT_MAPS:
        a = np.load(npy.out / export_name(stem, m))
        text = (other.out / export_name(stem, m, ext=ext)).read_text()
        want = "".join(sep.join(f"{float(v):.8g}" for v in row) + "\n" for row in a)
        assert text == want, m


@pytest.mark.slow
def test_cli_reports_a_missing_volume_on_stderr_and_still_exits_0(run_cli):
    """--trace retrace on a one-volume file: message, no files, exit 0 (A5)."""
    run = run_cli("arh5_retrace")
    assert run.result.exit_code == 0
    assert run.result.stdout == ""
    assert run.result.stderr == (
        f"Unhandled error processing {run.path}.\n"
        "list index out of range\n"
        "Continuing...\n"
    )
    assert written(run) == []


@pytest.mark.slow
def test_cli_as_a_subprocess_exits_0_with_verbose_paths_on_stdout(
    real_data_file, tmp_path
):
    path = Path(real_data_file(NID_RELPATH))
    stem = NID_STEM
    out = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    proc = subprocess.run(
        [sys.executable, "-m", "magic_afm.cli", "--output-path", str(out)]
        + ["--verbose", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        f"Creating {out}",
        f"Writing {out / options_name(stem)}",
        *(f"Writing {out / export_name(stem, m)}" for m in DEFAULT_MAPS),
    ]
    # stderr holds only tqdm's two progress bars, the file bar drawn last
    last = proc.stderr.replace("\r", "\n").rstrip().splitlines()[-1]
    assert last.startswith("Files completed: 100%")
    assert "| 1/1 [" in last
    assert f"Fitting {path.name} force curves: 100%" in proc.stderr
    assert "| 100/100 [" in proc.stderr
    for word in ("Traceback", "Unhandled", "Warning"):
        assert word not in proc.stderr
