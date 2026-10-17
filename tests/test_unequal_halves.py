"""Force curves whose extend and retract halves differ in length resample and fit."""

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

import numpy as np
import pytest
import test_goldens as t2
from test_fit_force_curve_signature import gui_options

from magic_afm import calculation
from magic_afm.calculation import RESAMPLE_NPTS, FitMode

# Cuts of the 500 + 500 point Demo curve: (extend slice, retract slice, stride). The
# retract stops early, or the extend starts late, so the turnaround stays in place.
UNEQUAL_CUTS = {
    "long_ext_resampled": (slice(None), slice(None, 350), 1),  # 500 + 350
    "long_ret_resampled": (slice(100, None), slice(None), 1),  # 400 + 500
    "long_ext_short": (slice(None), slice(None, 300), 2),  # 250 + 150
    "long_ret_short": (slice(100, None), slice(None), 2),  # 200 + 250
}


def unequal_curve(cut):
    """A Demo curve cut to unequal halves, ((zext, zret), (dext, dret))."""
    ext, ret, stride = UNEQUAL_CUTS[cut]
    (zext, zret), (dext, dret) = t2.demo_curve(**t2.SYNTHETIC_CASES["demo_r0_c0"])
    zxr = (zext[ext][::stride].copy(), zret[ret][::stride].copy())
    dxr = (dext[ext][::stride].copy(), dret[ret][::stride].copy())
    return zxr, dxr


def prepare(curve, mode):
    zxr, dxr = curve
    curve = tuple(map(np.copy, zxr)), tuple(map(np.copy, dxr))
    return calculation.prepare_force_curve(((0, 0), curve), fit_mode=mode, s_ratio=1.0)


def today_prepare_force_curve(x, fit_mode, s_ratio):
    """prepare_force_curve as it was before it took unequal halves, frozen."""
    rc, (zxr_and_dxr) = x
    npts = sum(map(len, zxr_and_dxr[0]))
    if npts > RESAMPLE_NPTS:
        zxr_and_dxr = np.reshape(zxr_and_dxr, (2, -1))
        zxr_and_dxr = calculation.resample_wrapper(zxr_and_dxr, RESAMPLE_NPTS, True)
        zxr_and_dxr = zxr_and_dxr.reshape(2, 2, -1)
    zxr, dxr = zxr_and_dxr

    if fit_mode == FitMode.EXTEND:
        z, d, split = zxr[0], dxr[0], None
    elif fit_mode == FitMode.RETRACT:
        z, d, split = zxr[1], dxr[1], None
    elif fit_mode == FitMode.BOTH:
        (z, d), split = np.reshape(zxr_and_dxr, (2, -1)), len(zxr_and_dxr[0][0])
    else:
        raise ValueError("Unknown fit_mode: ", fit_mode)
    d *= s_ratio
    return z, d, split, rc


@pytest.mark.parametrize("cut", [c for c in UNEQUAL_CUTS if c.endswith("resampled")])
def test_unequal_halves_resample_to_npts_split_at_the_scaled_turnaround(cut):
    curve = unequal_curve(cut)
    n_ext, n_ret = map(len, curve[0])
    npts = n_ext + n_ret
    assert npts > RESAMPLE_NPTS
    z_ext, d_ext, split_ext, _ = prepare(curve, FitMode.EXTEND)
    z_ret, d_ret, split_ret, _ = prepare(curve, FitMode.RETRACT)
    z, d, split, _ = prepare(curve, FitMode.BOTH)
    assert split_ext is None and split_ret is None
    assert len(z_ext) == len(d_ext) and len(z_ret) == len(d_ret)
    assert len(z_ext) + len(z_ret) == RESAMPLE_NPTS
    assert len(z) == len(d) == RESAMPLE_NPTS
    assert abs(split - n_ext * RESAMPLE_NPTS / npts) <= 1
    assert split == len(z_ext)
    assert (len(z_ext) > len(z_ret)) == (n_ext > n_ret)


@pytest.mark.parametrize("cut", [c for c in UNEQUAL_CUTS if c.endswith("resampled")])
def test_unequal_halves_resample_as_one_continuous_curve(cut):
    """The halves concatenated equal the resampled concatenation of the input."""
    curve = unequal_curve(cut)
    zxr, dxr = curve
    whole = np.stack([np.concatenate(zxr), np.concatenate(dxr)])
    want = calculation.resample_wrapper(whole, RESAMPLE_NPTS, True)
    z_ext, d_ext, _, _ = prepare(curve, FitMode.EXTEND)
    z_ret, d_ret, _, _ = prepare(curve, FitMode.RETRACT)
    z, d, _, _ = prepare(curve, FitMode.BOTH)
    for got in (np.stack([z, d]), np.stack([[*z_ext, *z_ret], [*d_ext, *d_ret]])):
        np.testing.assert_allclose(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("cut", [c for c in UNEQUAL_CUTS if c.endswith("short")])
def test_unequal_halves_at_or_below_npts_pass_through_unresampled(cut):
    curve = unequal_curve(cut)
    (zext, zret), (dext, dret) = curve
    assert len(zext) + len(zret) <= RESAMPLE_NPTS
    want = {
        FitMode.EXTEND: (zext, dext, None),
        FitMode.RETRACT: (zret, dret, None),
        FitMode.BOTH: (
            np.concatenate([zext, zret]),
            np.concatenate([dext, dret]),
            len(zext),
        ),
    }
    for mode, (want_z, want_d, want_split) in want.items():
        z, d, split, _ = prepare(curve, mode)
        np.testing.assert_allclose(z, want_z, rtol=0, atol=0, err_msg=mode.name)
        np.testing.assert_allclose(d, want_d, rtol=0, atol=0, err_msg=mode.name)
        assert split == want_split, mode.name


@pytest.mark.parametrize("stride", [1, 2], ids=["resampled", "short"])
@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
def test_equal_halves_prepare_exactly_as_before(mode, stride):
    (zext, zret), (dext, dret) = t2.demo_curve(**t2.SYNTHETIC_CASES["demo_r0_c0"])
    curve = (zext[::stride], zret[::stride]), (dext[::stride], dret[::stride])
    z, d, split, rc = prepare(curve, mode)
    zxr, dxr = curve
    frozen = (tuple(map(np.copy, zxr)), tuple(map(np.copy, dxr)))
    want_z, want_d, want_split, want_rc = today_prepare_force_curve(
        ((0, 0), frozen), mode, 1.0
    )
    np.testing.assert_allclose(z, want_z, rtol=0, atol=0)
    np.testing.assert_allclose(d, want_d, rtol=0, atol=0)
    assert split == want_split
    assert rc == want_rc


@pytest.mark.parametrize("cut", list(UNEQUAL_CUTS))
@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
def test_gui_readout_fits_an_unequal_half_curve(mode, cut):
    from magic_afm.gui import _impl as gui

    zxr, dxr = unequal_curve(cut)
    n_ext, n_ret = map(len, zxr)
    npts = n_ext + n_ret
    options = gui_options(mode, t2.DEMO_K)
    fvparams = gui.ForceVolumeParams(
        k=t2.DEMO_K, defl_sens=5.0, sync_dist=0, trace=None
    )
    with np.errstate(all="ignore"):
        data = gui.calculate_force_data(zxr, dxr, 1e-3, options, fvparams)
    out_ext, out_ret = map(len, data.zxr)
    assert tuple(map(len, data.dxr)) == (out_ext, out_ret)
    assert tuple(map(len, data.txr)) == (out_ext, out_ret)
    if npts > RESAMPLE_NPTS:
        assert out_ext + out_ret == RESAMPLE_NPTS
        assert abs(out_ext - n_ext * RESAMPLE_NPTS / npts) <= 1
    else:
        assert (out_ext, out_ret) == (n_ext, n_ret)
    assert (out_ext > out_ret) == (n_ext > n_ret)
    assert np.all(np.isfinite(data.parms.item()))


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
def test_gui_readout_resamples_equal_halves_exactly_as_before(mode):
    from magic_afm.gui import _impl as gui

    zxr, dxr = t2.demo_curve(**t2.SYNTHETIC_CASES["demo_r0_c0"])
    assert sum(map(len, zxr)) > RESAMPLE_NPTS
    frozen = (tuple(map(np.copy, zxr)), tuple(map(np.copy, dxr)))
    want_ext = today_prepare_force_curve(((0, 0), frozen), FitMode.EXTEND, 1.0)
    frozen = (tuple(map(np.copy, zxr)), tuple(map(np.copy, dxr)))
    want_ret = today_prepare_force_curve(((0, 0), frozen), FitMode.RETRACT, 1.0)
    options = gui_options(mode, t2.DEMO_K)
    fvparams = gui.ForceVolumeParams(
        k=t2.DEMO_K, defl_sens=5.0, sync_dist=0, trace=None
    )
    with np.errstate(all="ignore"):
        data = gui.calculate_force_data(zxr, dxr, 1e-3, options, fvparams)
    for got, want in zip(data.zxr, (want_ext[0], want_ret[0]), strict=True):
        np.testing.assert_allclose(got, want, rtol=0, atol=0)
    for got, want in zip(data.dxr, (want_ext[1], want_ret[1]), strict=True):
        np.testing.assert_allclose(got, want, rtol=0, atol=0)
