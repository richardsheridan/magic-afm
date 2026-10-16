"""Stopped scans: curves the file never acquired are NaN in every exported map."""

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

# hard0001 was stopped scanning up (rows 0-18 acquired) and hard0003 scanning down (rows
# 13-31), as the reader indexes them; each lacks 13 full rows of 32. The missing pixels
# are derived from what the reader yields. Acquired pixels are checked against a direct
# calculation-layer fit (the helpers in tests/test_goldens.py), so a change that blanked
# them too would fail.

import contextlib
import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import test_goldens as t2
import trio
import trio_parallel
from click.testing import CliRunner
from test_cli import (
    DEFAULT_MAPS,
    EXPORT_FIELDS,
    export_name,
    invoke_cli,
    map_tolerance,
    options_name,
)
from tqdm.std import tqdm as tqdm_std

from magic_afm.calculation import FitFix, FitMode

STOPPED_SCANS = {
    "hard0001": "test_cases/PDMS_SAA/ffm/hard0001.ARDF",
    "hard0003": "test_cases/PDMS_SAA/ffm/hard0003.ARDF",
}
MISSING_ROWS = 13


def acquired_and_fitted(path):
    """Mask of the curves the reader yields and their fitted maps, NaN elsewhere."""
    fvfile_cls, data = t2.open_fvfile(path)
    with data:
        fvfile = fvfile_cls.parse(data)
        volume = fvfile.volumes[0]  # the default trace selection
        acquired = np.zeros(volume.shape, dtype=bool)
        maps = {
            group: {name: np.full(volume.shape, np.nan) for name in names}
            for group, names in t2.FIELDS.items()
        }
        for rc_curve in volume.iter_curves():
            rc = rc_curve[0]
            assert not acquired[rc], rc
            acquired[rc] = True
            pre = t2.preprocess(rc_curve, FitMode.RETRACT)
            fitted = t2.fit_one(pre, FitMode.RETRACT, fvfile.k)
            for group, values in fitted.items():
                for name, value in values.items():
                    maps[group][name][rc] = value
    rows_missing = ~acquired.any(axis=1)
    assert rows_missing.sum() == MISSING_ROWS
    assert not acquired[rows_missing].any() and acquired[~rows_missing].all()
    return acquired, maps


def assert_nan_exactly_where_not_acquired(maps, acquired, want_maps):
    """Every map matches want on acquired pixels and is all-NaN on the rest."""
    for m, a in maps.items():
        group, field, factor = EXPORT_FIELDS[m]
        want = factor * want_maps[group][field]
        rtol, atol = map_tolerance(m)
        np.testing.assert_array_equal(
            np.isnan(a[acquired]), np.isnan(want[acquired]), err_msg=m
        )
        np.testing.assert_allclose(
            a[acquired], want[acquired], rtol=rtol, atol=atol, err_msg=m
        )
    not_nan = {m: int((~np.isnan(a[~acquired])).sum()) for m, a in maps.items()}
    missing = int((~acquired).sum())
    assert not_nan == dict.fromkeys(maps, 0), f"not NaN of {missing} never acquired"


@pytest.mark.slow
@pytest.mark.parametrize("stem", STOPPED_SCANS)
def test_cli_maps_are_nan_exactly_at_curves_a_stopped_scan_never_acquired(
    real_data_file, tmp_path, stem
):
    path = Path(real_data_file(STOPPED_SCANS[stem]))
    acquired, want_maps = acquired_and_fitted(path)
    out = tmp_path / "out"
    result = invoke_cli(path, out, ())
    assert result.exit_code == 0
    expected = [export_name(stem, m) for m in DEFAULT_MAPS] + [options_name(stem)]
    assert sorted(p.name for p in out.iterdir()) == sorted(expected)
    # the CLI exports rows reversed relative to the reader
    maps = {
        m: np.load(out / export_name(stem, m)).astype(np.float64)[::-1]
        for m in DEFAULT_MAPS
    }
    assert_nan_exactly_where_not_acquired(maps, acquired, want_maps)


class RecordingProgressBar(tqdm_std):
    """The GUI's tqdm_tk without Tk (or the CLI's tqdm): a tqdm writing to a buffer."""

    instances: list["RecordingProgressBar"] = []

    def __init__(self, *args, **kwargs):
        for tk_only in ("tk_parent", "grab", "cancel_callback", "pause_callback"):
            kwargs.pop(tk_only, None)
        super().__init__(*args, file=io.StringIO(), **kwargs)
        self.instances.append(self)


def run_gui_fit(path, monkeypatch, shape):
    """Run the GUI's property-map fit with no Tk display; return its bar and file."""
    from magic_afm.gui import _impl as gui

    monkeypatch.setattr(gui, "tqdm_tk", RecordingProgressBar)
    monkeypatch.setattr(RecordingProgressBar, "instances", [])
    monkeypatch.setattr(trio_parallel, "run_sync", trio.to_thread.run_sync)

    fvfile_cls, data = t2.open_fvfile(path)
    with data:
        opened_fvol = gui.GUIFVFile(fvfile_cls.parse(data))
        params = opened_fvol.initial_parameters
        controller = gui.ForceVolumeController(MagicMock(), opened_fvol)
        controller.axesimage = MagicMock(**{"get_size.return_value": shape})
        controller.spinner_scope = contextlib.nullcontext
        options = gui.ForceCurveOptions(
            fit_mode=FitMode.RETRACT,
            fit_fix=FitFix.DEFAULTS,
            disp_kind=gui.DispKind.zd,
            k=params.k,
            k_sens=True,
            defl_sens=params.defl_sens,
            sync_dist=params.sync_dist,
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
        with np.errstate(all="ignore"):
            trio.run(controller.calc_prop_map_response, options)
    (pbar,) = RecordingProgressBar.instances
    return pbar, opened_fvol


@pytest.mark.slow
@pytest.mark.parametrize("stem", STOPPED_SCANS)
def test_gui_maps_are_nan_exactly_at_curves_a_stopped_scan_never_acquired(
    real_data_file, monkeypatch, stem
):
    path = Path(real_data_file(STOPPED_SCANS[stem]))
    acquired, want_maps = acquired_and_fitted(path)
    _, opened_fvol = run_gui_fit(path, monkeypatch, acquired.shape)
    # the GUI keeps the reader's row order
    maps = {
        m: np.asarray(opened_fvol.get_image("CalcRet" + m), dtype=np.float64)
        for m in DEFAULT_MAPS
    }
    assert_nan_exactly_where_not_acquired(maps, acquired, want_maps)


def shape_and_yielded_count(path):
    """The default volume's grid shape and how many curves its reader yields."""
    fvfile_cls, data = t2.open_fvfile(path)
    with data:
        volume = fvfile_cls.parse(data).volumes[0]
        shape = volume.shape
        yielded = sum(1 for _ in volume.iter_indices())
    assert yielded == shape[1] * (shape[0] - MISSING_ROWS)
    return shape, yielded


@pytest.mark.slow
@pytest.mark.parametrize("stem", STOPPED_SCANS)
def test_gui_progress_total_is_the_number_of_curves_the_reader_yields(
    real_data_file, monkeypatch, stem
):
    path = Path(real_data_file(STOPPED_SCANS[stem]))
    shape, yielded = shape_and_yielded_count(path)
    pbar, _ = run_gui_fit(path, monkeypatch, shape)
    assert pbar.total == pbar.n == yielded


@pytest.mark.slow
@pytest.mark.parametrize("stem", STOPPED_SCANS)
def test_cli_progress_total_is_the_number_of_curves_the_reader_yields(
    real_data_file, monkeypatch, tmp_path, stem
):
    from magic_afm.cli import _impl as cli

    path = Path(real_data_file(STOPPED_SCANS[stem]))
    _, yielded = shape_and_yielded_count(path)
    monkeypatch.setattr(cli, "tqdm", RecordingProgressBar)
    monkeypatch.setattr(RecordingProgressBar, "instances", [])
    # progress stays enabled: a disabled tqdm never counts
    argv = ["--output-path", str(tmp_path / "out"), "--stop-on-error", str(path)]
    result = CliRunner().invoke(cli.main, argv, catch_exceptions=False)
    assert result.exit_code == 0
    (pbar,) = (
        bar for bar in RecordingProgressBar.instances if bar.desc.startswith("Fitting")
    )
    assert pbar.total == pbar.n == yielded


def open_stopped(real_data_file, stem):
    """The parsed file, its open data and the mask of curves the reader yields."""
    fvfile_cls, data = t2.open_fvfile(Path(real_data_file(STOPPED_SCANS[stem])))
    fvfile = fvfile_cls.parse(data)
    volume = fvfile.volumes[0]
    acquired = np.zeros(volume.shape, dtype=bool)
    for rc in volume.iter_indices():
        acquired[rc] = True
    return fvfile, data, acquired


@pytest.mark.slow
@pytest.mark.parametrize("stem", STOPPED_SCANS)
def test_get_curve_is_nan_of_the_curve_length_where_a_stopped_scan_never_acquired(
    real_data_file, stem
):
    fvfile, data, acquired = open_stopped(real_data_file, stem)
    with data:
        volume = fvfile.volumes[0]
        (azext, azret), (adext, adret) = volume.get_curve(*np.argwhere(acquired)[0])
        for r, c in np.argwhere(~acquired).tolist():
            (zext, zret), (dext, dret) = volume.get_curve(r, c)
            for half, like in zip(
                (zext, zret, dext, dret), (azext, azret, adext, adret), strict=True
            ):
                assert np.shape(half) == np.shape(like)
                assert np.isnan(half).all()


@pytest.mark.slow
@pytest.mark.parametrize("stem", STOPPED_SCANS)
def test_images_are_nan_where_a_stopped_scan_never_acquired(real_data_file, stem):
    fvfile, data, acquired = open_stopped(real_data_file, stem)
    with data:
        for name, image in fvfile.images.items():
            assert np.isnan(image.get_image()[~acquired]).all(), name
