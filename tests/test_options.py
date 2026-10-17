"""Options JSON reading: bad input is a ValueError naming the key and a CLI usage
error, and the JSONs the released betas wrote still load."""

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

import io
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from click.testing import CliRunner
from synthetic_files.nanoscope import build_pfc

from magic_afm import calculation
from magic_afm._options import TraceChoice, dump_options, load_options
from magic_afm.calculation import FitFix, FitMode
from magic_afm.cli._impl import main

OLD_JSON_DIR = Path(__file__).parent / "fixtures" / "options_json"

# (key the message must name, options JSON text)
BAD_VALUE_CASES = {
    "fit_mode_bad_name": ("fit_mode", '{"fit_mode": "NOPE"}'),
    "trace_bad_name": ("trace", '{"trace": "NOPE"}'),
    "radius_null": ("radius", '{"radius": null}'),
    "fit_mode_null": ("fit_mode", '{"fit_mode": null}'),
    "radius_list": ("radius", '{"radius": [1]}'),
    "radius_bad_string": ("radius", '{"radius": "abc"}'),
    "fit_fix_bad_string": ("fit_fix", '{"fit_fix": "x"}'),
    "fit_fix_bad_name": ("fit_fix", '{"fit_fix": ["TAU", "NOPE"]}'),
    "resample_npts_bad_string": ("resample_npts", '{"resample_npts": "x"}'),
    "schema_version_newer": ("schema_version", '{"schema_version": 2}'),
    "unknown_key": ("bogus", '{"bogus": 1}'),
}

# options JSON text with no key to name
BAD_SHAPE_CASES = {
    "malformed": "{bad",
    "top_level_list": "[1, 2]",
    "top_level_null": "null",
}
NOT_OBJECT_CASES = {"top_level_list", "top_level_null"}


@pytest.mark.parametrize(
    ("key", "text"), BAD_VALUE_CASES.values(), ids=BAD_VALUE_CASES.keys()
)
def test_load_options_bad_value_raises_valueerror_naming_key(key, text):
    with pytest.raises(ValueError) as excinfo:
        load_options(io.StringIO(text))
    assert key in str(excinfo.value)


@pytest.mark.parametrize("case", BAD_SHAPE_CASES)
def test_load_options_bad_shape_raises_valueerror(case):
    with pytest.raises(ValueError) as excinfo:
        load_options(io.StringIO(BAD_SHAPE_CASES[case]))
    if case in NOT_OBJECT_CASES:
        assert "object" in str(excinfo.value)


def _run_cli_with_options_json(tmp_path, text):
    options_json = tmp_path / "options.json"
    options_json.write_text(text)
    # exists with a known suffix, so only the options JSON can be rejected
    data_file = tmp_path / "curves.ARDF"
    data_file.touch()
    return CliRunner().invoke(
        main, ["--options-json", str(options_json), str(data_file)]
    )


def _error_lines(output):
    return [line for line in output.splitlines() if line.startswith("Error:")]


@pytest.mark.parametrize(
    ("key", "text"), BAD_VALUE_CASES.values(), ids=BAD_VALUE_CASES.keys()
)
def test_cli_bad_options_json_value_is_usage_error_naming_key(tmp_path, key, text):
    result = _run_cli_with_options_json(tmp_path, text)
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    error_lines = _error_lines(result.output)
    assert len(error_lines) == 1, result.output
    assert key in error_lines[0]


@pytest.mark.parametrize("case", BAD_SHAPE_CASES)
def test_cli_bad_options_json_shape_is_usage_error(tmp_path, case):
    result = _run_cli_with_options_json(tmp_path, BAD_SHAPE_CASES[case])
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert len(_error_lines(result.output)) == 1, result.output


# What today's load_options decodes from the JSONs the betas wrote (see
# tests/fixtures/options_json/README.md). These pin current behaviour, including the
# read-compat cases a later options schema change must handle:
# - fit_fix is an int in every fixture (123, FitFix.DEFAULTS).
# - trace is an int from the GUI (1) but a name from the CLI ("TRACE").
# - the 1.0.0b2 CLI default M is 1e9 (meant as Pa) and now reads as 1e9 GPa;
#   the 1.0.0b2 GUI wrote 10 ** (6 - 9), already GPa; 1.0.0b3 wrote 1.0 GPa.
# - the 1.0.0b2 CLI wrote no k_sens key; the GUI and 1.0.0b3 did.
_FIT_DEFAULTS = dict(
    radius=20.0,
    tau=0.0,
    lj_scale=2.0,
    vd=0.0,
    li_per=0.0,
    li_amp=0.0,
    drag=0.0,
    fit_fix=FitFix.DEFAULTS,
    fit_mode=FitMode.RETRACT,
)
_PFC = dict(k=0.46, defl_sens=35.0, sync_dist=69.6625)
OLD_JSON_VALUES = {
    "1.0.0b2-cli": {**_FIT_DEFAULTS, **_PFC, "trace": None, "M": 1e9},
    "1.0.0b2-cli-trace": {
        **_FIT_DEFAULTS,
        **_PFC,
        "trace": TraceChoice.TRACE,
        "M": 1e9,
    },
    "1.0.0b2-gui-pfc": {
        **_FIT_DEFAULTS,
        **_PFC,
        "trace": None,
        "M": 0.001,
        "k_sens": True,
    },
    "1.0.0b2-gui-ardf": {
        **_FIT_DEFAULTS,
        "k": 0.462,
        "defl_sens": 55.471000000000004,
        "sync_dist": None,
        "trace": TraceChoice.TRACE,
        "M": 0.001,
        "k_sens": True,
    },
    "1.0.0b3-cli": {**_FIT_DEFAULTS, **_PFC, "trace": None, "M": 1.0, "k_sens": True},
}
ENUM_TYPES = {"trace": TraceChoice, "fit_fix": FitFix, "fit_mode": FitMode}


def test_every_old_json_fixture_is_checked():
    assert {p.stem for p in OLD_JSON_DIR.glob("*.json")} == OLD_JSON_VALUES.keys()


@pytest.mark.parametrize("name", OLD_JSON_VALUES)
def test_load_options_reads_json_written_by_released_beta(name):
    with (OLD_JSON_DIR / (name + ".json")).open() as fp:
        options = load_options(fp)
    assert options == OLD_JSON_VALUES[name]
    for key, enum_type in ENUM_TYPES.items():
        if options[key] is not None:
            assert type(options[key]) is enum_type, key


# M is in GPa, so an unversioned file whose M is above 1e4 GPa (10 TPa, about ten times
# diamond) can only hold a value meant as Pa, such as the 1.0.0b2 CLI default of 1e9.
M_WARNING_FIXTURES = {"1.0.0b2-cli", "1.0.0b2-cli-trace"}


@pytest.mark.parametrize("name", OLD_JSON_VALUES)
def test_load_options_warns_naming_m_only_for_the_old_cli_default(name):
    with (OLD_JSON_DIR / (name + ".json")).open() as fp:
        if name in M_WARNING_FIXTURES:
            with pytest.warns(Warning, match=r"\bM\b"):
                load_options(fp)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                load_options(fp)


@pytest.mark.parametrize("M", [1e9, 1e5])
def test_load_options_unversioned_implausible_m_warns_naming_m(M):
    with pytest.warns(Warning, match=r"\bM\b"):
        options = load_options(io.StringIO(json.dumps({"M": M})))
    assert options["M"] == M


@pytest.mark.parametrize("M", [1.0, 0.001, 1.2e3])
def test_load_options_unversioned_real_modulus_does_not_warn(M):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        options = load_options(io.StringIO(json.dumps({"M": M})))
    assert options["M"] == M


def test_load_options_accepts_schema_version_1():
    options = load_options(io.StringIO('{"schema_version": 1, "radius": 20.0}'))
    assert options["radius"] == 20.0


def test_load_options_reads_fit_fix_as_list_of_member_names():
    options = load_options(io.StringIO('{"fit_fix": ["TAU", "VD"]}'))
    assert options["fit_fix"] == FitFix.TAU | FitFix.VD
    assert type(options["fit_fix"]) is FitFix


@pytest.mark.parametrize("value", [0, 5, 123])
def test_load_options_reads_old_integer_fit_fix(value):
    options = load_options(io.StringIO(json.dumps({"fit_fix": value})))
    assert options["fit_fix"] == FitFix(value)
    assert type(options["fit_fix"]) is FitFix


def full_options(**changes):
    """A complete options dict, as a front end hands it to dump_options."""
    options = {
        **_FIT_DEFAULTS,
        **_PFC,
        "trace": None,
        "M": 1.0,
        "k_sens": True,
        "resample_npts": calculation.RESAMPLE_NPTS,
    }
    return {**options, **changes}


FIT_FIX_NAMES = {
    "tau_vd": (FitFix.TAU | FitFix.VD, ["TAU", "VD"]),
    "defaults": (
        FitFix.DEFAULTS,
        ["RADIUS", "TAU", "VD", "LI_PER", "LI_AMP", "DRAG"],
    ),
    "none": (FitFix(0), []),
}


@pytest.mark.parametrize(
    ("fit_fix", "names"), FIT_FIX_NAMES.values(), ids=FIT_FIX_NAMES.keys()
)
def test_dump_options_writes_fit_fix_as_set_member_names(fit_fix, names):
    written = json.loads(dump_options(full_options(fit_fix=fit_fix)))
    assert written["fit_fix"] == names


def test_dump_options_writes_schema_version_1_and_resample_npts():
    written = json.loads(dump_options(full_options(resample_npts=64)))
    assert written["schema_version"] == 1
    assert written["resample_npts"] == 64


@pytest.mark.parametrize("trace", [None, TraceChoice.TRACE])
@pytest.mark.parametrize("fit_fix", [v[0] for v in FIT_FIX_NAMES.values()])
def test_dump_options_round_trips_through_load_options(fit_fix, trace):
    original = full_options(fit_fix=fit_fix, trace=trace, resample_npts=300)
    loaded = load_options(io.StringIO(dump_options(original)))
    assert {k: loaded[k] for k in original} == original
    for key, enum_type in ENUM_TYPES.items():
        if loaded[key] is not None:
            assert type(loaded[key]) is enum_type, key


class RecordingExecutor(ThreadPoolExecutor):
    """Runs the fits in threads and records the length of each curve fitted."""

    def __init__(self, *args, **kwargs):
        # drop the pool's initializer, which would renice this process
        super().__init__(max_workers=1)
        self.lengths = []
        RecordingExecutor.instances.append(self)

    def submit(self, fn, chunk, *args, **kwargs):
        self.lengths.extend(len(z) for z, d, split, rc in chunk[1])
        return super().submit(fn, chunk, *args, **kwargs)


def run_cli_recording_fits(tmp_path, monkeypatch, options):
    """Run the CLI on a 2 x 2 capture of 256-point curves with an options JSON."""
    from magic_afm.cli import _impl as cli

    monkeypatch.setattr(RecordingExecutor, "instances", [], raising=False)
    monkeypatch.setattr(cli, "ProcessPoolExecutor", RecordingExecutor)
    path = build_pfc(tmp_path, (2, 2), points=128).path
    options_json = tmp_path / "options.json"
    options_json.write_text(json.dumps(options))
    out = tmp_path / "out"
    argv = ["--output-path", str(out), "--disable-progress"]
    argv += ["--options-json", str(options_json), str(path)]
    result = CliRunner().invoke(main, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    (executor,) = RecordingExecutor.instances
    (written,) = out.glob("*_options*.json")
    return executor.lengths, json.loads(written.read_text())


def test_cli_resamples_to_resample_npts_from_options_json(tmp_path, monkeypatch):
    lengths, written = run_cli_recording_fits(
        tmp_path, monkeypatch, {"fit_mode": "BOTH", "resample_npts": 64}
    )
    assert lengths == [64] * 4
    assert written["resample_npts"] == 64


def test_cli_resample_npts_defaults_to_resample_npts_constant(tmp_path, monkeypatch):
    lengths, written = run_cli_recording_fits(
        tmp_path, monkeypatch, {"fit_mode": "BOTH"}
    )
    assert calculation.RESAMPLE_NPTS > 256
    assert lengths == [256] * 4
    assert written["resample_npts"] == calculation.RESAMPLE_NPTS
    assert written["schema_version"] == 1


def test_gui_export_writes_versioned_options_json(tmp_path, monkeypatch):
    import tkinter.filedialog
    import tkinter.messagebox

    import matplotlib
    import trio

    from magic_afm.gui import _impl as gui

    def no_dialog(**kwargs):
        raise RuntimeError(kwargs.get("message"))

    fname = tmp_path / "export.npy"
    monkeypatch.setattr(
        tkinter.filedialog, "asksaveasfilename", lambda **kwargs: str(fname)
    )
    monkeypatch.setattr(tkinter.messagebox, "showerror", no_dialog)
    monkeypatch.setitem(matplotlib.rcParams, "savefig.directory", "")
    options = gui.ForceCurveOptions(
        fit_mode=FitMode.RETRACT,
        fit_fix=FitFix.TAU | FitFix.VD,
        disp_kind=gui.DispKind.zd,
        k=0.46,
        k_sens=True,
        defl_sens=35.0,
        sync_dist=None,
        radius=20.0,
        tau=0.0,
        M=1.0,
        lj_scale=2.0,
        vd=0.0,
        li_per=0.0,
        li_amp=0.0,
        drag=0.0,
        trace=None,
    )
    navbar_cls = type(
        "HeadlessNavbar",
        (),
        {"_aexport_calculations": gui.AsyncNavigationToolbar2Tk._aexport_calculations},
    )
    navbar = navbar_cls()
    navbar._get_options = lambda: options
    navbar._get_image_names = lambda: []
    navbar._get_image_by_name = None
    navbar._prev_filename = ""
    trio.run(navbar._aexport_calculations)

    written = json.loads((tmp_path / "export_options.json").read_text())
    assert written["schema_version"] == 1
    assert written["fit_fix"] == ["TAU", "VD"]
    assert written["resample_npts"] == calculation.RESAMPLE_NPTS
    with (tmp_path / "export_options.json").open() as fp:
        assert load_options(fp)["fit_fix"] == FitFix.TAU | FitFix.VD
