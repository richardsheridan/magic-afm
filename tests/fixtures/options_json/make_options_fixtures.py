"""Write the options JSON fixtures as the released betas wrote them (plan item T3a).

The old code runs from git worktrees checked out at the tags, on the test
environment of this checkout. From the repository root:

    git worktree add --detach WT_B2 1.0.0b2
    git worktree add --detach WT_B3 1.0.0b3
    pixi run -e test python tests/fixtures/options_json/make_options_fixtures.py \\
        --b2 WT_B2 --b3 WT_B3 --out OUTDIR
    git worktree remove WT_B2; git worktree remove WT_B3

then compare OUTDIR with this directory and copy the files over byte for byte
(never re-dump them). pytest never runs this. The input files come from
example.data (or MAFM_TEST_DATA) and are unpacked into a temporary directory.

Each run is a separate interpreter with PYTHONPATH set to the worktree, so the
tag's own magic_afm is imported; the child asserts that it is. The CLI runs
are the tag's own click entry point. The GUI JSON at 1.0.0b2 is written by the
tag's own AsyncNavigationToolbar2Tk._aexport_calculations (the "Export
calculated images" button), with its file dialog patched to return a path and
no images to export. It reads the options through the tag's own
ForceVolumeTkDisplay.options property, on a stand-in display whose Tk variables
hold the widget defaults from ForceVolumeTkDisplay.__init__ and _add_parm, with
"Retract" selected. The variables are real tkinter variables on a Tcl-only
interpreter, so no window opens.
"""

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

# The CLI's workers are spawned, and spawn re-runs this file as __mp_main__
# in each of them, so everything but the definitions sits under main().

import argparse
import lzma
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

PFC = "test_cases/PS_PCL_SAA/QNM/onekhz.000.pfc"
ARDF = "test_cases/PDMS_SAA/ffm/both0000.ARDF"

# fixture name: (tag, front end, input relpath, extra CLI args)
RUNS = {
    "1.0.0b2-cli": ("b2", "cli", PFC, []),
    "1.0.0b2-cli-trace": ("b2", "cli", PFC, ["--trace", "trace"]),
    "1.0.0b2-gui-pfc": ("b2", "gui", PFC, None),
    "1.0.0b2-gui-ardf": ("b2", "gui", ARDF, None),
    "1.0.0b3-cli": ("b3", "cli", PFC, []),
}

# 1.0.0b2's fit fails on numpy >= 2.5 (float() of a 1-element array); b53b6a0,
# released in 1.0.0b3, fixed it. Its calculation.py hunk is applied to the b2
# worktree. It touches the fit only, not the JSON writers.
SHIMS = {"b2": ["b53b6a0"]}


def data_root():
    return Path(os.environ.get("MAFM_TEST_DATA") or REPO_ROOT / "example.data")


def unpack(relpath, dest_dir):
    dest = Path(dest_dir) / Path(relpath).name
    if dest.exists():
        return dest
    raw = data_root() / relpath
    if raw.is_file():
        shutil.copyfile(raw, dest)
    else:
        with lzma.open(str(raw) + ".xz") as inp, dest.open("wb") as out:
            shutil.copyfileobj(inp, out)
    return dest


def check_tag_import(worktree):
    import magic_afm

    here = Path(magic_afm.__file__).resolve()
    assert here.is_relative_to(Path(worktree).resolve()), here


def child_cli(worktree, args):
    check_tag_import(worktree)
    from magic_afm.cli._impl import main as cli_main

    cli_main(args)


def child_gui(worktree, path, outdir):
    check_tag_import(worktree)
    import tkinter
    import tkinter.filedialog
    import tkinter.messagebox
    import warnings

    import trio

    from magic_afm import calculation, data_readers
    from magic_afm.gui import _impl as g

    path = Path(path)
    fvfile_cls, opener = data_readers.SUFFIX_FVFILE_MAP[path.suffix.lower()]
    with opener(path) as open_thing:
        fvfile = fvfile_cls.parse(open_thing)
        initial_values = g.AsyncFVFile(fvfile).initial_parameters

    tcl = tkinter.Tcl()  # Tk variables without Tk: no window

    def strvar(value):
        var = tkinter.StringVar(tcl)
        var.set(value)  # as sbox.set(default) in _add_parm
        return var

    FitFix, FitMode = calculation.FitFix, calculation.FitMode
    display_cls = type(
        "HeadlessDisplay",
        (),
        {
            k: g.ForceVolumeTkDisplay.__dict__[k]
            for k in (
                "options",
                "_read_chkboxes",
                "get_sync_dist_or_none",
                "get_trace_or_none",
            )
        },
    )
    d = display_cls()
    if initial_values.trace is not None:
        d.data_select_intvar = tkinter.IntVar(tcl, value=initial_values.trace)
    d.disp_kind_intvar = tkinter.IntVar(tcl, value=g.DispKind.zd.value)
    d.defl_sens_strvar = strvar(initial_values.defl_sens)
    d.spring_const_strvar = strvar(initial_values.k)
    if initial_values.sync_dist is not None:
        d.sync_dist_strvar = strvar(initial_values.sync_dist)
    # default SKIP; the user selects Retract before calculating and exporting
    d.fit_intvar = tkinter.IntVar(tcl, value=FitMode.RETRACT.value)
    d.calc_k_sens_intvar = tkinter.IntVar(tcl, value=True)
    # _add_parm calls in order: modulus and radius share one "fix radius"
    # variable (keyed 0 and RADIUS), the rest default to FitFix.DEFAULTS
    radius_var = tkinter.IntVar(tcl, True)
    d.chkboxes = {0: (radius_var, None), FitFix.RADIUS: (radius_var, None)}
    for flag in (
        FitFix.TAU,
        FitFix.LJ_SCALE,
        FitFix.VD,
        FitFix.DRAG,
        FitFix.LI_PER,
        FitFix.LI_AMP,
    ):
        d.chkboxes[flag] = (tkinter.IntVar(tcl, bool(flag & FitFix.DEFAULTS)), None)
    d.mod_strvar = strvar(6.0)
    d.radius_strvar = strvar(20.0)
    d.tau_strvar = strvar(0.0)
    d.lj_scale_strvar = strvar(2)
    d.vd_strvar = strvar(0.0)
    d.drag_strvar = strvar(0.0)
    d.li_per_strvar = strvar(0.0)
    d.li_amp_strvar = strvar(0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # options turns its errors into warnings
        opts = d.options
    assert opts.fit_fix == FitFix.DEFAULTS, opts.fit_fix  # GUI's fitfix_intvar

    fname = str(Path(outdir) / (path.name + ".npy"))

    def no_dialog(**kwargs):
        raise RuntimeError(kwargs.get("message"))

    tkinter.filedialog.asksaveasfilename = lambda **kwargs: fname
    tkinter.messagebox.showerror = no_dialog
    navbar_cls = type(
        "HeadlessNavbar",
        (),
        {"_aexport_calculations": (g.AsyncNavigationToolbar2Tk._aexport_calculations)},
    )
    navbar = navbar_cls()
    navbar._get_options = lambda: d.options  # as in ForceVolumeController
    navbar._get_image_names = lambda: []
    navbar._get_image_by_name = None
    navbar._prev_filename = ""
    trio.run(navbar._aexport_calculations)


def apply_shims(worktree, commits):
    """Apply later upstream fixes to the fitting code only, once."""
    for commit in commits:
        diff = subprocess.run(
            ["git", "-C", REPO_ROOT, "show", "--format=", commit, "--"]
            + ["magic_afm/calculation.py"],
            check=True,
            capture_output=True,
        ).stdout
        git_apply = ["git", "-C", worktree, "apply", "-C1"]
        applied = subprocess.run(
            git_apply + ["--reverse", "--check"], input=diff, capture_output=True
        )
        if applied.returncode:
            subprocess.run(git_apply, input=diff, check=True)


def run_child(worktree, *args):
    env = {**os.environ, "PYTHONPATH": str(worktree)}
    cmd = [sys.executable, "-P", __file__, "--child", str(worktree), *map(str, args)]
    subprocess.run(cmd, env=env, check=True, cwd=tempfile.gettempdir())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--b2", type=Path, required=True, help="worktree at 1.0.0b2")
    parser.add_argument("--b3", type=Path, required=True, help="worktree at 1.0.0b3")
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)
    for tag, commits in SHIMS.items():
        apply_shims(getattr(ns, tag), commits)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, (tag, front, relpath, extra) in RUNS.items():
            worktree = getattr(ns, tag)
            path = unpack(relpath, tmp)
            outdir = tmp / name
            outdir.mkdir()
            if front == "cli":
                args = ["cli", "--disable-progress", "--stop-on-error"]
                args += ["--output-path", outdir, *extra, path]
                run_child(worktree, *args)
            else:
                run_child(worktree, "gui", path, outdir)
            (written,) = outdir.glob("*_options*.json")
            shutil.copyfile(written, ns.out / (name + ".json"))
            print(name, "<-", written.name)


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        worktree, kind, *rest = sys.argv[i + 1 :]
        if kind == "cli":
            child_cli(worktree, rest)
        else:
            child_gui(worktree, *rest)
    else:
        main()
