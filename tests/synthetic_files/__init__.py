"""Small synthetic data files in each vendor format, for the reader tests.

One module per format holds a ``build_*`` function. Each takes the directory to
write into, the grid shape ``(rows, cols)``, the points per curve and the vendor
metadata that matter (spring constant, deflection sensitivity, scan size, units),
writes one file laid out the way the vendor software writes it, and returns a
SyntheticFile: the path plus what a reader should get back, as physical
values. Curves are in nm, as the readers return them. Values the vendor stores
as scaled integers are given as the file decodes to them, in float64, so a
reader that computes in float32 matches to float32 rounding, not bit for bit.

The builders use only numpy and h5py, never magic_afm, so that they describe the
formats rather than the readers; ``open_fvfile`` is the one helper here that
calls the reader, through its public surface.
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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Index = tuple[int, int]
HalfCurves = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SyntheticVolume:
    """Force curves on a (rows, cols) grid, in nm, each half in time order.

    ``zext`` and ``dext`` run from far from the surface to near it, ``zret`` and
    ``dret`` from near to far. Curves may differ in length: each array is
    ``(rows, cols, n)`` with n the longest curve, padded with NaN past that
    curve's own ``ext_points`` or ``ret_points``. ``order`` lists the pixels in
    the order the file stores their curves, or is None when the format does not
    fix one.
    """

    name: str
    zext: np.ndarray
    zret: np.ndarray
    dext: np.ndarray
    dret: np.ndarray
    ext_points: np.ndarray
    ret_points: np.ndarray
    order: tuple[Index, ...] | None = None

    @property
    def shape(self) -> Index:
        rows, cols = self.ext_points.shape
        return rows, cols

    def curve(
        self, r: int, c: int, n_ext: int | None = None, n_ret: int | None = None
    ) -> tuple[HalfCurves, HalfCurves]:
        """Curve (r, c) as ((zext, zret), (dext, dret)).

        With n_ext or n_ret, only that many points of the half nearest the
        surface: the end of the extend half and the start of the retract half.
        """
        ne, nr = int(self.ext_points[r, c]), int(self.ret_points[r, c])
        n_ext = ne if n_ext is None else n_ext
        n_ret = nr if n_ret is None else n_ret
        if not (0 < n_ext <= ne and 0 < n_ret <= nr):
            raise ValueError("curve is shorter than asked", (r, c), (ne, nr))
        ext = np.s_[r, c, ne - n_ext : ne]
        ret = np.s_[r, c, :n_ret]
        return (
            (self.zext[ext], self.zret[ret]),
            (self.dext[ext], self.dret[ret]),
        )


@dataclass(frozen=True)
class SyntheticFile:
    """A file a builder wrote and what a reader should get from it.

    ``k`` in N/m, ``defl_sens`` in nm/V, ``t_step`` in s per point, ``scansize``
    in nm along the fast then the slow axis. ``volumes`` are in the order the
    reader lists them. ``images`` are (rows, cols) maps in the units the file
    stores them in, named as the reader names them, with their units in
    ``image_units`` where the file states one. ``units`` holds the units the
    file declares for its raw channels and spring constant. ``sync_dist`` is how
    many stored points the deflection lags the drive, for formats that store
    the deflection of an oscillating drive and not its z, else None.
    """

    path: Path
    volumes: list[SyntheticVolume]
    images: dict[str, np.ndarray]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]
    image_units: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    sync_dist: float | None = None


def point_counts(shape: Index, points) -> np.ndarray:
    """Points of each curve on the grid, from one int or a (rows, cols) array."""
    counts = np.broadcast_to(np.asarray(points, dtype=np.int64), shape).copy()
    if counts.min() < 1:
        raise ValueError("every curve needs at least one point", counts.min())
    return counts


def force_curves(
    rng: np.random.Generator,
    ext_points: np.ndarray,
    ret_points: np.ndarray,
    *,
    z_far: float = 0.0,
    ramp: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plausible approach and retract curves in nm, different on every pixel.

    The extend half rises by ``ramp`` over the longest extend curve from about
    ``z_far``, so a shorter curve stops nearer the start, as a force-triggered
    ramp does; the retract half goes back from where its extend half stopped.
    Deflection is flat off the surface, dips on snap-in, rises linearly past a
    per-pixel contact point and on the way back sticks below zero until it pulls
    off. Returns zext, zret, dext, dret, NaN past each curve's own length.
    """
    shape = ext_points.shape
    n_ext, n_ret = int(ext_points.max()), int(ret_points.max())
    start = z_far + rng.uniform(0.0, ramp / 20, shape)[..., None]
    contact = start + rng.uniform(0.4, 0.6, shape)[..., None] * ramp
    slope = rng.uniform(0.2, 0.8, shape)[..., None]
    snap_in = rng.uniform(1.0, 3.0, shape)[..., None]
    pull_off = rng.uniform(5.0, 20.0, shape)[..., None]

    step = ramp / max(n_ext - 1, 1)
    zext = start + step * np.arange(n_ext)
    turn = np.take_along_axis(zext, ext_points[..., None] - 1, axis=-1)
    zret = turn - ramp / max(n_ret - 1, 1) * np.arange(n_ret)

    dext = slope * np.maximum(zext - contact, 0.0)
    # wide enough to hold at least one point however coarse the ramp
    snap_window = (zext < contact) & (zext >= contact - max(5.0, step))
    dext = np.where(snap_window, -snap_in, dext)
    dret = slope * (zret - contact)
    dret = np.where(dret > -pull_off, dret, 0.0)
    dext += rng.normal(0.0, 0.05, dext.shape)
    dret += rng.normal(0.0, 0.05, dret.shape)

    for z, d, n in ((zext, dext, ext_points), (zret, dret, ret_points)):
        past_end = np.arange(z.shape[-1]) >= n[..., None]
        z[past_end] = np.nan
        d[past_end] = np.nan
    return zext, zret, dext, dret


@contextmanager
def open_fvfile(path: Path) -> Iterator:
    """Parse path the way the CLI does: SUFFIX_FVFILE_MAP's class and opener."""
    from magic_afm.data_readers import SUFFIX_FVFILE_MAP

    fvfile_cls, opener = SUFFIX_FVFILE_MAP[path.suffix.lower()]
    with opener(path) as data:
        yield fvfile_cls.parse(data)
