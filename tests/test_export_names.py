"""Exported property maps: the GUI and the CLI name, filter and scale them alike."""

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

# Per-map files are {root}_{Ext|Ret|Both}{Name}[Err]{suffix}.{ext}, the suffix being
# the fitted volume's name on a file holding both a Trace and a Retrace volume and
# empty otherwise, and the options file is {root}_options.json. The root is the data
# file's stem in the CLI and the save-dialog name in the GUI; here both are the stem.

import json

import numpy as np
import pytest
from synthetic_files.ardf import build_ardf
from synthetic_files.nanoscope import build_pfc
from test_cli import DEFAULT_MAPS, export_name, invoke_cli, options_name
from test_stopped_scans import run_gui_fit

SHAPE = (4, 4)
POINTS = 128  # per half curve; at 32 none of the synthetic curves fit


def build(directory, capture):
    """A synthetic ARDF holding the trace, the retrace or both volumes."""
    filename = f"{capture}.ARDF"
    return build_ardf(directory, SHAPE, POINTS, capture=capture, filename=filename).path


def expected_names(stem, *, mode="Ret", suffix=""):
    maps = [export_name(stem, m, mode=mode, trace=suffix) for m in DEFAULT_MAPS]
    return sorted([*maps, options_name(stem)])


def written(out):
    return sorted(p.name for p in out.iterdir())


def gui_export(opened_fvol, fname, monkeypatch, trace):
    """Run the GUI's export of its Calc images, the save dialog answering fname."""
    import tkinter.filedialog
    import tkinter.messagebox

    import matplotlib
    import trio

    from magic_afm.gui import _impl as gui

    def no_dialog(**kwargs):
        raise RuntimeError(kwargs.get("message"))

    monkeypatch.setattr(
        tkinter.filedialog, "asksaveasfilename", lambda **kwargs: str(fname)
    )
    monkeypatch.setattr(tkinter.messagebox, "showerror", no_dialog)
    monkeypatch.setitem(matplotlib.rcParams, "savefig.directory", "")
    navbar_cls = type(
        "HeadlessNavbar",
        (),
        {"_aexport_calculations": gui.AsyncNavigationToolbar2Tk._aexport_calculations},
    )
    navbar = navbar_cls()
    navbar._get_options = lambda: gui.ForceCurveOptions(
        fit_mode=gui.calculation.FitMode.RETRACT,
        fit_fix=gui.calculation.FitFix.DEFAULTS,
        disp_kind=gui.DispKind.zd,
        k=opened_fvol.initial_parameters.k,
        k_sens=True,
        defl_sens=opened_fvol.initial_parameters.defl_sens,
        sync_dist=opened_fvol.initial_parameters.sync_dist,
        radius=20.0,
        tau=0.0,
        M=1.0,
        lj_scale=2.0,
        vd=0.0,
        li_per=0.0,
        li_amp=0.0,
        drag=0.0,
        trace=trace,
    )
    navbar._get_image_names = lambda: list(opened_fvol.image_names)
    navbar._get_image_by_name = opened_fvol.get_image
    navbar._prev_filename = ""
    trio.run(navbar._aexport_calculations)


# case: (capture, CLI arguments, GUI trace selection, expected suffix). The GUI's
# default on a two-direction file is the Trace radio button, initial_parameters.trace.
FRONT_END_CASES = {
    "both_default": ("both", (), 1, "Trace"),
    "both_retrace": ("both", ("--trace", "retrace"), 0, "Retrace"),
    "one_direction": ("trace", (), None, ""),
}


@pytest.mark.parametrize("case", FRONT_END_CASES)
def test_gui_and_cli_export_the_same_file_names_and_arrays(tmp_path, monkeypatch, case):
    capture, args, gui_trace, suffix = FRONT_END_CASES[case]
    path = build(tmp_path, capture)
    stem = path.stem
    cli_out, gui_out = tmp_path / "cli", tmp_path / "gui"
    result = invoke_cli(path, cli_out, args)
    assert result.exit_code == 0, result.output
    _, opened_fvol = run_gui_fit(path, monkeypatch, SHAPE, trace=gui_trace)
    gui_out.mkdir()
    gui_export(opened_fvol, gui_out / f"{stem}.npy", monkeypatch, gui_trace)

    assert written(gui_out) == expected_names(stem, suffix=suffix)
    assert written(cli_out) == written(gui_out)
    modulus = np.load(cli_out / export_name(stem, "IndentationModulus", trace=suffix))
    assert np.isfinite(modulus).any()  # the comparison below is not of NaN alone
    for m in DEFAULT_MAPS:
        name = export_name(stem, m, trace=suffix)
        a, b = np.load(gui_out / name), np.load(cli_out / name)
        assert a.dtype == b.dtype == np.float32, m
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True, err_msg=m)


# case: (file, CLI arguments, expected suffix)
CLI_SUFFIX_CASES = {
    "both_default": ("both", (), "Trace"),
    "both_trace": ("both", ("--trace", "trace"), "Trace"),
    "both_retrace": ("both", ("--trace", "retrace"), "Retrace"),
    "trace_only_default": ("trace", (), ""),
    "trace_only_trace": ("trace", ("--trace", "trace"), ""),
    "retrace_only_retrace": ("retrace", ("--trace", "retrace"), ""),
    "pfc_trace": ("pfc", ("--trace", "trace"), ""),
}


@pytest.mark.parametrize("case", CLI_SUFFIX_CASES)
def test_cli_suffix_is_the_volume_name_only_on_a_two_direction_file(tmp_path, case):
    capture, args, suffix = CLI_SUFFIX_CASES[case]
    if capture == "pfc":
        path = build_pfc(tmp_path, SHAPE, filename="undirected.000.pfc").path
    else:
        path = build(tmp_path, capture)
    out = tmp_path / "out"
    result = invoke_cli(path, out, args)
    assert result.exit_code == 0, result.output
    assert written(out) == expected_names(path.stem, suffix=suffix)


def test_cli_two_fit_modes_into_one_directory_share_one_options_file(tmp_path):
    """Accepted: a second fit mode into the same directory overwrites the options."""
    path = build(tmp_path, "both")
    out = tmp_path / "out"
    for mode in ("retract", "extend"):
        result = invoke_cli(path, out, ("--fit-mode", mode))
        assert result.exit_code == 0, result.output
    ret = expected_names(path.stem, mode="Ret", suffix="Trace")
    ext = expected_names(path.stem, mode="Ext", suffix="Trace")
    assert written(out) == sorted(set(ret) | set(ext))
    options = json.loads((out / options_name(path.stem)).read_text())
    assert options["fit_mode"] == "EXTEND"
