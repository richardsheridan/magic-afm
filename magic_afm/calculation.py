"""Magic AFM Calculation

This module contains the actual, core code to calculate indentation ratios and
modulus sensitivities from force curve data, and is self-sufficient from the
rest of the Magic AFM package. If you want to replicate the results from our
paper or inspect your own force curves for "Magic Ratio" conditions, this
module should contain everything you need.
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
import enum
import traceback

import numpy as np
from numpy.linalg import lstsq

from soxr import resample

try:
    from numba import jit
except ImportError:
    jit = lambda *a, **kw: (lambda x: x)
else:
    abs = np.fabs


EPS = float(np.finfo(np.float64).eps)
RT_EPS = float(np.sqrt(EPS))
RESAMPLE_NPTS = 512

gkern = np.array([0.25, 0.5, 0.25], dtype=np.float32)


PROPERTY_UNITS_DICT = {
    "IndentationModulus": "Pa",
    "AdhesionForce": "N",
    "IndentationModulusErr": "Pa",
    "AdhesionForceErr": "N",
    "Deflection": "m",
    "Indentation": "m",
    "ContactRadius": "m",
    "TrueHeight": "m",  # Hi z -> lo h
    "IndentationRatio": None,
    "SensIndMod_k": None,
    "SumSquaredError": None,
}

PROPERTY_DTYPE = np.dtype([(name, "f4") for name in PROPERTY_UNITS_DICT])


@jit(nopython=True, nogil=True, cache=True)
def gauss3x3(img):
    img = np.copy(img)
    rows, cols = np.shape(img)
    tmp = np.empty(cols + 6, dtype=img.dtype)
    for row in img:
        tmp[:3] = row[0]
        tmp[3:-3] = row
        tmp[-3:] = row[-1]
        row[:] = np.convolve(tmp, gkern)[4:-4]
    if rows != cols:
        tmp = np.empty(rows + 6, dtype=img.dtype)
    for col in img.transpose():
        tmp[:3] = col[0]
        tmp[3:-3] = col
        tmp[-3:] = col[-1]
        col[:] = np.convolve(tmp, gkern)[4:-4]
    return img


@jit(nopython=True, nogil=True, cache=True)
def median3x1(img):
    out = np.empty_like(img)
    rows, cols = np.shape(out)
    for r in range(rows):
        if r == 0:
            vstart, vstop = (0, 2)
        elif r == rows - 1:
            vstart, vstop = (r - 1, rows)
        else:
            vstart, vstop = (r - 1, r + 2)
        for c in range(cols):
            out[r, c] = np.median(img[vstart:vstop, c])
    return out


@jit(nopython=True, nogil=True, cache=True)
def median3x3(img):
    out = np.empty_like(img)
    rows, cols = np.shape(out)
    for r in range(rows):
        if r == 0:
            vstart, vstop = (0, 2)
        elif r == rows - 1:
            vstart, vstop = (r - 1, rows)
        else:
            vstart, vstop = (r - 1, r + 2)
        for c in range(cols):
            if c == 0:
                hstart, hstop = (0, 2)
            elif c == cols - 1:
                hstart, hstop = c - 1, cols
            else:
                hstart, hstop = c - 1, c + 2
            out[r, c] = np.median(img[vstart:vstop, hstart:hstop])
    return out


@jit(nopython=True, nogil=True, cache=True)
def fillnan(img):
    out = np.copy(img)
    rows, cols = np.shape(out)
    nan_inds = np.nonzero(np.isnan(img))
    for r, c in zip(*nan_inds):
        if r == 0:
            vstart, vstop = (0, 2)
        elif r == rows - 1:
            vstart, vstop = (r - 1, rows)
        else:
            vstart, vstop = (r - 1, r + 2)
        if c == 0:
            hstart, hstop = (0, 2)
        elif c == cols - 1:
            hstart, hstop = c - 1, cols
        else:
            hstart, hstop = c - 1, c + 2
        out[r, c] = np.nanmedian(img[vstart:vstop, hstart:hstop])
    return out


def flatten(img):
    a = np.vander(np.arange(np.shape(img)[1]), 2)
    b = img.T
    # noinspection PyUnresolvedReferences
    keep = ~np.logical_or.reduce(~np.isfinite(b), axis=1)
    try:
        x = lstsq(a[keep, :], b[keep, :], rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full_like(img, fill_value=np.nan)
    return img - (a @ x).T


def planefit(img):
    ind = np.indices(np.shape(img))
    # [1, x, y]
    a = np.stack((np.ones_like(img.ravel()), ind[0].ravel(), ind[1].ravel())).T
    b = img.ravel()
    keep = np.isfinite(b)
    try:
        x = lstsq(a[keep, :], b[keep], rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full_like(img, fill_value=np.nan)
    return img - (a @ x).reshape(img.shape)


def offset(img):
    return img - np.nanmin(img)


MANIPULATIONS = dict(
    (
        ("Flatten", flatten),
        ("PlaneFit", planefit),
        ("Offset", offset),
        ("Median3x1", median3x1),
        ("Median3x3", median3x3),
        ("Gauss3x3", gauss3x3),
        ("FillNaNs", fillnan),
    )
)


def resample_wrapper(X, npts, fourier=True, restore_trend=True):
    """Resample individual z-d curves with Fourier or linear interpolation"""
    X = np.atleast_2d(X)
    old_npts = X.shape[-1]
    if fourier:
        X = X.T  # resample needs (frames, channels) i.e. (npts, features)
        trend = X[:5].mean(axis=0, keepdims=True)
        X = X - trend
        X = resample(X, old_npts, npts, "LQ")
        if restore_trend:
            X += trend
        return X.T
    else:
        tnew = np.linspace(0, 1, npts, endpoint=False)
        told = np.linspace(0, 1, old_npts, endpoint=False)
        return np.stack([np.interp(tnew, told, x) for x in X])


# can't cache because UUID cache busting https://github.com/numba/numba/issues/6284
@jit(nopython=True, nogil=True)
def secant(func, args, x0, x1):
    """Secant method from scipy optimize but stripping np.isclose for speed

    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    p0 = x0 * 1.0
    tol = RT_EPS
    maxiter = 50
    if x1 is not None:
        if x1 == x0:
            raise ValueError("x1 and x0 must be different")
        p1 = x1 * 1.0
    else:
        eps = RT_EPS
        p1 = x0 * (1 + eps)
        p1 += eps if p1 >= 0 else -eps
    q0 = func(p0, *args)
    q1 = func(p1, *args)
    if abs(q1) < abs(q0):
        p0, p1, q0, q1 = p1, p0, q1, q0
    for itr in range(maxiter):
        if q1 == q0:
            p = (p1 + p0) / 2.0
            break
        else:
            if abs(q1) > abs(q0):
                p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
            else:
                p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
        if abs(p - p1) <= tol:
            break
        p0, q0 = p1, q1
        p1 = p
        q1 = func(p1, *args)
    else:
        p = (p1 + p0) / 2.0

    return p.real


# can't cache because UUID cache busting https://github.com/numba/numba/issues/6284
# noinspection PyUnboundLocalVariable
@jit(nopython=True, nogil=True)
def brentq(func, args, xa, xb):
    """Transliterated from SciPy Zeros/brentq.c

    :returns x where func(x,*args) == 0 or None if the bracket does not cross 0

    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    xtol = RT_EPS
    maxiter = 50
    xpre = xa
    xcur = xb

    fpre = func(xpre, *args)
    if fpre == 0:
        return xpre

    fcur = func(xcur, *args)
    if fcur == 0:
        return xcur

    if fpre * fcur > 0:
        return None

    for i in range(maxiter):
        if fpre * fcur < 0:
            # always true on first iteration
            # don't worry about potential NameErrors
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < xtol:
            return xcur

        if abs(spre) > xtol and abs(fcur) < abs(fpre):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = (
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                )

            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - xtol):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > xtol:
            xcur += scur
        else:
            xcur += xtol if sbis > 0 else -xtol

        fcur = func(xcur, *args)

    return xcur


@jit(nopython=True, nogil=True, cache=True)
def mylinspace(start, stop, num, endpoint):
    """np.linspace is surprisingly intensive, so trim the fat

    Copyright (c) 2005-2020, NumPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above
           copyright notice, this list of conditions and the following
           disclaimer in the documentation and/or other materials provided
           with the distribution.

        * Neither the name of the NumPy Developers nor the names of any
           contributors may be used to endorse or promote products derived
           from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    delta = stop - start
    div = (num - 1) if endpoint else num

    y = np.arange(0.0, num)
    y *= delta / div
    y += start
    if endpoint and num > 1:
        y[-1] = stop
    return y


from ._vendored_lstsq import leastsq


def curve_fit(function, xdata, ydata, p0, sigma=None, bounds=None):
    """Wrap to match api of scipy.optimize.curve_fit

    This is faster (?!) and has no scipy dependency."""
    if bounds is None:
        constraints = None
    else:
        constraints = []
        for lo, hi in bounds.T.tolist():
            if lo == 0.0 and hi == np.inf:
                constraints.append((1, None, None))
            else:
                constraints.append((2, lo, hi))
    return leastsq(function, xdata, ydata, p0, sigma, constraints, full_output=True)


@jit(nopython=True, nogil=True, cache=True)
def schwarz_red(red_f, red_fc, stable, offset):
    """Calculate Schwarz potential indentation depth from force in reduced units.

    Stable indicates whether to calculate the stable or unstable branch of the solution.
    1 -> stable, -1 -> unstable

    Offset is useful for root finding routines to solve for a specific value of delta.
    """
    # if red_f<red_fc, it is likely a small numerical error
    # this fixes the issue for arrays and python scalars without weird logic
    df = abs(red_f - red_fc)

    # Save computations if pure DMT
    if red_fc == -2:
        return df ** (2 / 3) - offset

    # fmt: off
    red_contact_radius = (
        (3 * red_fc + 6) ** (1 / 2) + stable * df ** (1 / 2)
    ) ** (2 / 3)

    red_delta = red_contact_radius ** 2 - 4 * (
        red_contact_radius * (red_fc + 2) / 3
    ) ** (1 / 2)
    # fmt: on

    return red_delta - offset


@jit(nopython=True, nogil=True, cache=True)
def schwarz_wrap(red_force, red_fc, red_k, lj_delta_scale):
    """So that schwarz can be directly jammed into delta_curve"""
    return schwarz_red(red_force, red_fc, 1.0, 0.0)


# Exponents for LJ potential
bigpow = 9
lilpow = 3
powrat = bigpow / lilpow

# algebraically solved LJ parameters such that the minimum value is at delta=1
prefactor = powrat ** (1 / (bigpow - lilpow))
postfactor = 1 / (prefactor ** (-bigpow) - prefactor ** (-lilpow))  # such that lj(1)=1
# delta of minimum slope
lj_limit_factor = ((bigpow + 1) / (lilpow + 1)) ** (1 / (bigpow - lilpow))


@jit(nopython=True, nogil=True, cache=True)
def lj_force(delta, delta_scale, force_scale, delta_offset, force_offset):
    """Calculate a leonard-Jones force curve.

    Prefactor scaled such that minimum slope is at delta=1/delta_scale"""
    nondim_position = (delta - delta_offset) / delta_scale
    # np.divide is a workaround for nondim_position=0 so that ZeroDivisionError -> inf
    attraction = np.divide(1, (prefactor * nondim_position) ** lilpow)
    return postfactor * force_scale * (attraction**powrat - attraction) - force_offset


@jit(nopython=True, nogil=True, cache=True)
def lj_gradient(delta, delta_scale, force_scale, delta_offset, force_offset):
    """Gradient of lj_force.

    Offset is useful for root finding (reduced spring constant matching)"""
    nondim_position = (delta - delta_offset) / delta_scale
    # np.divide is a workaround for nondim_position=0 so that ZeroDivisionError -> inf
    attraction = np.divide(
        lilpow * prefactor ** (-lilpow), nondim_position ** (lilpow + 1)
    )
    repulsion = bigpow * prefactor ** (-bigpow) * nondim_position ** (-bigpow - 1)
    return (
        postfactor * force_scale * (attraction - repulsion) / delta_scale - force_offset
    )


@jit(nopython=True, nogil=True, cache=True)
def interp_with_offset(x, xp, fp, offset):
    return np.interp(x, xp, fp) - offset


@jit(nopython=True, nogil=True, cache=True)
def mygradient(f, d):
    """np.gradient is also surprisingly intensive, trim to 1d case

    Copyright (c) 2005-2020, NumPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above
           copyright notice, this list of conditions and the following
           disclaimer in the documentation and/or other materials provided
           with the distribution.

        * Neither the name of the NumPy Developers nor the names of any
           contributors may be used to endorse or promote products derived
           from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    dx = np.diff(d)
    uniform_spacing = (dx == dx[0]).all()
    out = np.empty_like(f)

    # Numerical differentiation: 2nd order interior
    if uniform_spacing:
        out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx[0])
    else:
        dx1 = dx[0:-1]
        dx2 = dx[1:]
        a = -dx2 / (dx1 * (dx1 + dx2))
        b = (dx2 - dx1) / (dx1 * dx2)
        c = dx1 / (dx2 * (dx1 + dx2))
        out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]

    # Numerical differentiation: 1st order edges
    out[0] = (f[1] - f[0]) / dx[0]
    out[-1] = (f[-1] - f[-2]) / dx[-1]

    return out


def red_extend(red_delta, red_fc, red_k, lj_delta_scale):
    """Calculate, in reduced units, an extent Schwarz curve with long range LJ potential and snap-off physics.

    MUCH LESS TESTED THAN RED_RETRACT

    Following the broad example of Lin (2007) and Johnson (1997), we combine the stable branch of the
    Schwarz potential with a long range Leonard-Jones potential via finite-k snap-off physics.

    To do this, we build two curves, one for each physical model. The LJ potential is clipped
    above where its slope equals the reduced lever spring constant and replaced with a tangent line.
    The Schwarz potential is clipped at the intersection point and replaced with negative infinities.
    Doing so makes the higher of the two curves at all points give the result.

    Or in pithy math form we want
    maximum(NINF⌢S, LJ⌢snap)
    """

    # The indentation at the critical force must be calculated directly, no shortcuts
    red_deltac = schwarz_red(red_fc, red_fc, 1.0, 0.0)

    # lj_force() outputs a curve with minimum at (lj_delta_scale,lj_force_scale)
    # offset needed to put minimum at red_deltac
    lj_delta_offset = red_deltac - lj_delta_scale
    lj_f = lj_force(red_delta, lj_delta_scale, red_fc, lj_delta_offset, 0.0)
    lj_f = np.atleast_1d(lj_f)  # Later parts can choke on scalars

    # Try to find where LJ gradient == red_k
    # brentq is marginally faster with python scalars without jit
    red_d_min = float(np.min(red_delta))
    args = (lj_delta_scale, red_fc, lj_delta_offset, red_k)
    bracket = (
        red_d_min,
        lj_limit_factor * lj_delta_scale + lj_delta_offset,
    )
    lj_end_pos = brentq(
        lj_gradient,
        args,
        *bracket,
    )
    if lj_end_pos is None:
        # If root finding fails, put the end pos somewhere quasi-reasonable
        if lj_gradient(red_d_min, *args) <= 0:
            lj_end_pos = red_d_min
        else:
            lj_end_pos = lj_delta_scale + lj_delta_offset

    lj_end_force = lj_force(lj_end_pos, lj_delta_scale, red_fc, lj_delta_offset, 0.0)

    # Overwrite after end position with tangent line
    lj_f[red_delta >= lj_end_pos] = (
        red_delta[red_delta >= lj_end_pos] - lj_end_pos
    ) * red_k + lj_end_force

    # Unfortunately Schwarz is d(f) rather than f(d) so we need to infer the largest possible force
    red_d_max = np.max(red_delta)
    red_fc, red_d_max = float(red_fc), float(red_d_max)
    args = red_fc, 1.0, red_d_max
    red_f_max = secant(schwarz_red, args, x0=0.0, x1=red_fc)

    # because of noise in force channel, need points well beyond red_f_max
    # XXX: a total hack, would be nice to have a real stop and num for this linspace
    f = mylinspace(red_fc, 1.5 * (red_f_max - red_fc) + red_fc, 100, True)
    d = schwarz_red(f, red_fc, 1.0, 0.0)

    s_f = np.interp(red_delta, d, f, left=-np.inf)

    return np.maximum(s_f, lj_f)


def red_retract(red_delta, red_fc, red_k, lj_delta_scale):
    """Calculate, in reduced units, a retract Schwarz curve with long range LJ potential and snap-off physics.

    Following the broad example of Lin (2007) and Johnson (1997), we combine the unstable branch
    of the Schwarz potential with a long range Leonard-Jones potential via finite-k snap-off physics.

    To do this, we build two curves, one for each physical model. The unstable Schwarz branch is clipped
    below where its slope equals the reduced lever spring constant and replaced with a tangent line.
    The LJ potential is clipped at it's minimum (positioned by definition at (red_deltac, red_fc)) and
    replaced with infinities. Doing so makes the lower of the two curves at all points give the result.

    Or in pithy math form we want
    minimum(snap⌢S, LJ⌢INF)
    """
    # We can take a shortcut if precisely DMT physics since the unstable branch vanishes
    not_DMT = not red_fc == -2

    # The indentation at the critical force must be calculated directly, no shortcuts
    red_deltac = schwarz_red(red_fc, red_fc, 1.0, 0.0)

    # lj_force() outputs a curve with minimum at (lj_delta_scale,lj_force_scale)
    # offset needed to put minimum at red_deltac
    lj_delta_offset = red_deltac - lj_delta_scale
    lj_f = lj_force(red_delta, lj_delta_scale, red_fc, lj_delta_offset, 0.0)
    lj_f = np.atleast_1d(lj_f)  # Later parts can choke on scalars

    # make points definitely lose in minimum function if in the contact/stable branch
    lj_f[red_delta >= red_deltac] = np.inf

    # Unfortunately Schwarz is d(f) rather than f(d) so we need to infer the largest possible force
    red_d_max = np.max(red_delta)
    red_fc, red_d_max = float(red_fc), float(red_d_max)
    args = red_fc, 1.0, red_d_max
    red_f_max = secant(schwarz_red, args, x0=red_fc, x1=0.0)

    # because of noise in force channel, need points well beyond red_f_max
    # XXX: a total hack, would be nice to have a real stop and num for this linspace
    f = mylinspace(red_fc, 1.5 * (red_f_max - red_fc) + red_fc, 100, False)
    d = schwarz_red(f, red_fc, 1.0, 0.0)

    # Use this endpoint in DMT case or if there is a problem with the slope finding
    s_end_pos = 0
    s_end_force = -2

    if not_DMT:
        # Find slope == red_k between vertical and horizontal parts of unstable branch
        f0 = mylinspace((7 * red_fc + 8) / 3, red_fc, 100, endpoint=False)
        d0 = schwarz_red(f0, red_fc, -1.0, 0.0)
        df0dd0 = mygradient(f0, d0)

        s_end_pos = brentq(interp_with_offset, (d0, df0dd0, red_k), d0[0], d0[-1])
        if s_end_pos is not None:
            # found a good end point
            s_end_force = np.interp(s_end_pos, d0, f0)
            f = np.concatenate((f0, f))
            d = np.concatenate((d0, d))
        else:
            s_end_pos = 0

    s_f = np.interp(red_delta, d, f, left=np.inf, right=0)
    s_f = np.atleast_1d(s_f).squeeze()  # don't choke on weird array shapes or scalars

    # overwrite portion after end position with tangent line
    s_f[red_delta <= s_end_pos] = (
        red_delta[red_delta <= s_end_pos] - s_end_pos
    ) * red_k + s_end_force

    return np.minimum(s_f, lj_f)


def force_curve(
    red_curve, delta, k, radius, M, fc, tau, delta_shift, force_shift, lj_delta_scale
):
    """Calculate a force curve from indentation data."""
    # Catch crazy inputs early
    assert k > 0, k
    assert radius > 0, radius
    assert M > 0, M
    assert fc < 0, fc
    assert 0 <= tau <= 1, tau

    # Calculate reduced/dimensionless values of inputs
    # tau = tau1**2 in Schwarz => ratio of short range to total surface energy
    red_fc = (tau - 4) / 2
    ref_force = fc / red_fc
    ref_radius = (ref_force * radius / M) ** (1 / 3)
    ref_delta = ref_radius * ref_radius / radius
    red_delta = (delta - delta_shift) / ref_delta
    red_k = k / ref_force * ref_delta
    lj_delta_scale = np.exp(lj_delta_scale) / ref_delta

    # Match sign conventions of force curve calculations now rather than later
    red_force = red_curve(red_delta, red_fc, -red_k, -lj_delta_scale)

    # Rescale to dimensioned units
    return (red_force * ref_force) + force_shift


def delta_curve(
    red_curve, force, k, radius, M, fc, tau, delta_shift, force_shift, lj_delta_scale
):
    """Convenience function for inverse of force_curve, i.e. force->delta"""
    # Catch crazy inputs early
    assert k > 0, k
    assert radius > 0, radius
    assert M > 0, M
    assert fc < 0, fc
    assert 0 <= tau <= 1, tau

    # Calculate reduced/dimensionless values of inputs
    # tau = tau1**2 in Schwarz => ratio of short range to total surface energy
    red_fc = (tau - 4) / 2
    ref_force = fc / red_fc
    red_force = (force - force_shift) / ref_force
    ref_radius = (ref_force * radius / M) ** (1 / 3)
    ref_delta = ref_radius * ref_radius / radius
    red_k = k / ref_force * ref_delta
    lj_delta_scale = np.exp(lj_delta_scale) / ref_delta

    # Match sign conventions of force curve calculations now rather than later
    red_delta = red_curve(red_force, red_fc, -red_k, -lj_delta_scale)

    # Rescale to dimensioned units
    return (red_delta * ref_delta) + delta_shift


@enum.unique
class FitMode(enum.IntEnum):
    SKIP = 0  # needs to be inty and falsy
    EXTEND = 1
    RETRACT = 2
    BOTH = enum.auto()


def rapid_forcecurve_estimate(delta, force, radius):
    """Very coarse estimate of force curve parameters for fit initial guess"""

    fzero = np.median(force)

    imin = np.argmin(force)
    deltamin = delta[imin]

    # if adhesion is low, min may be noise; clip delta to middle of curve
    deltamid = (delta[0] + delta[-1]) / 2
    deltamin = max(deltamin, deltamid)

    fmin = force[imin]
    fc_guess = fzero - fmin

    imax = np.argmax(delta)
    deltamax = delta[imax]
    fmax = force[imax]
    M_guess = (fmax - fmin) / np.sqrt(radius * (deltamax - deltamin) ** 3)
    if not np.isfinite(M_guess):
        M_guess = 1.0

    return M_guess, fc_guess, deltamin, fzero, 1.0


def fitfun(
    delta, force, k, radius, tau, fit_mode, cancel_poller=bool, p0=None, **kwargs
):
    if p0 is None:
        p0 = rapid_forcecurve_estimate(delta, force, radius)

    bounds = np.transpose(
        (
            (0.0, np.inf),  # M
            (0.0, np.inf),  # fc
            # (0, 1),           # tau
            (np.min(delta), np.max(delta)),  # delta_shift
            (np.min(force), np.max(force)),  # force_shift
            (-6.0, 6.0),  # lj_delta_scale
        )
    )
    p0 = np.clip(p0, *bounds)

    assert fit_mode
    if fit_mode == FitMode.EXTEND:
        red_curve = red_extend
    elif fit_mode == FitMode.RETRACT:
        red_curve = red_retract
    elif fit_mode == FitMode.BOTH:
        split = kwargs["split"]

        def red_curve(delta, *args):
            return np.concatenate(
                (
                    red_extend(delta[:split], *args),
                    red_retract(delta[split:], *args),
                )
            )

    else:
        raise ValueError("Unknown fit_mode: ", fit_mode)

    def partial_force_curve(delta, M, fc, delta_shift, force_shift, lj_delta_scale):
        cancel_poller()
        if np.any(np.isnan((M, fc, delta_shift, force_shift, lj_delta_scale))):
            print("Fit likely failed: NaNs in params")
            return np.full_like(delta, np.nan)
        return force_curve(
            red_curve,
            delta,
            k,
            radius,
            M,
            -fc,
            tau,
            delta_shift,
            force_shift,
            lj_delta_scale,
        )

    try:
        beta, cov, infodict, *_ = curve_fit(
            partial_force_curve, delta, force, p0=p0, bounds=bounds
        )
        sse = infodict["chisq"]
        beta_err = infodict["uncertainties"]
    except Exception:
        traceback.print_exc()
        print(p0)
        beta = np.full_like(p0, np.nan)
        beta_err = beta
        sse = np.nan

    return beta, beta_err, sse, partial_force_curve


def calc_def_ind_ztru_ac(force, beta, radius, k, tau, fit_mode, **kwargs):
    """Calculate deflection, indentation, z_true_surface given deflection data and parameters."""
    M, fc, delta_shift, force_shift, lj_delta_scale, *_ = beta

    # sign convention mismatch
    fc = -fc

    assert fit_mode
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

    maxdelta = delta_curve(
        schwarz_wrap,
        maxforce,
        k,
        radius,
        M,
        fc,
        tau,
        delta_shift,
        force_shift,
        lj_delta_scale,
    )
    mindelta = delta_curve(
        schwarz_wrap,
        force_shift + fc,
        k,
        radius,
        M,
        fc,
        tau,
        delta_shift,
        force_shift,
        lj_delta_scale,
    )
    # Identical on extend or retract, but in the case of `FitMode.BOTH` need to pick one
    zeroindforce = float(
        force_curve(
            red_retract,
            delta_shift,
            k,
            radius,
            M,
            fc,
            tau,
            delta_shift,
            force_shift,
            lj_delta_scale,
        )
    )

    maxforce -= force_shift
    red_fc = (tau - 4) / 2
    ref_force = fc / red_fc
    df = abs(maxforce / ref_force - red_fc)
    red_contact_radius = ((3 * red_fc + 6) ** (1 / 2) + df ** (1 / 2)) ** (2 / 3)
    contact_radius = red_contact_radius * (M / ref_force / radius) ** (-1 / 3)

    deflection = (maxforce - fc) / k
    indentation = maxdelta - mindelta
    z_true_surface = delta_shift + zeroindforce / k
    return deflection, indentation, z_true_surface, mindelta, contact_radius


def perturb_k(delta, f, epsilon, k):
    k_new = (1 + epsilon) * k
    f_new = f * ((1 + epsilon) ** 0.5)
    delta_new = delta + (f - f_new / (1 + epsilon)) / k
    return delta_new, f_new, k_new


def calc_properties_imap(delta_f_rc, **kwargs):
    delta, force, split, rc = delta_f_rc
    kwargs["split"] = split
    beta, beta_err, sse, partial_force_curve = fitfun(delta, force, **kwargs)
    if np.any(np.isnan(beta)):
        return rc, None
    ind_mod = beta[0]
    adh_force = beta[1]
    ind_mod_err = beta_err[0]
    adh_force_err = beta_err[1]
    (deflection, indentation, z_true_surface, mindelta, a_c) = calc_def_ind_ztru_ac(
        force, beta, **kwargs
    )
    k = kwargs.pop("k")
    eps = 1e-3
    beta_perturb, *_ = fitfun(*perturb_k(delta, force, eps, k), p0=beta, **kwargs)
    if np.any(np.isnan(beta_perturb)):
        return rc, None
    ind_mod_perturb = beta_perturb[0]
    ind_mod_sens_k = (ind_mod_perturb - ind_mod) / ind_mod / eps

    # pack up properties, ensuring all fields are filled in order
    properties = np.void(np.nan, dtype=PROPERTY_DTYPE)
    properties["IndentationModulus"] = ind_mod * 1e9
    properties["AdhesionForce"] = adh_force / 1e9
    properties["IndentationModulusErr"] = ind_mod_err * 1e9
    properties["AdhesionForceErr"] = adh_force_err / 1e9
    properties["Deflection"] = deflection / 1e9
    properties["Indentation"] = indentation / 1e9
    properties["ContactRadius"] = a_c / 1e9
    properties["TrueHeight"] = -z_true_surface / 1e9
    properties["IndentationRatio"] = deflection / indentation
    properties["SensIndMod_k"] = ind_mod_sens_k
    properties["SumSquaredError"] = sse
    # .item() coerces structured dtype to tuple
    assert not np.any(np.isnan(properties.item()))

    return rc, properties


def warmup_jit():
    """Call jitted functions until check_jit output stabilizes"""
    npts = 1024
    split = npts // 2
    delta = np.float32(
        (np.cos(np.linspace(0, np.pi * 2, npts, endpoint=False)) - 0.90) * 25
    )

    fext = force_curve(red_extend, delta[:split], 1, 10, 1, -10, 1, 0, 0, 1)
    fret = force_curve(red_retract, delta[split:], 1, 10, 1, -10, 1, 0, 0, 1)
    fext = force_curve(red_extend, delta[:split], 1, 10, 1, -10, 0, 0, 0, 1)
    fret = force_curve(red_retract, delta[split:], 1, 10, 1, -10, 0, 0, 0, 1)
    maxforce = fret[slice(len(fret) // 25)].mean()
    maxdelta = delta_curve(schwarz_wrap, maxforce, 1, 10, 1, -10, 0, 0, 0, 1)
    maxdelta = delta_curve(schwarz_wrap, maxforce, 1, 10, 1, -10, 1, 0, 0, 1)
    image = np.zeros((64, 64), dtype=np.float32)
    gauss3x3(image)
    median3x1(image)
    fillnan(image)
    median3x3(image)


def check_jit():
    for _ in globals().values():
        if hasattr(_, "get_metadata"):
            print(_)
            print(_.get_metadata().keys())


def load_force_curve(opened_fvol, fit_mode, k, s_ratio, rc):
    x = opened_fvol.get_curve(*rc)
    return process_force_curve((rc, x), fit_mode, k, s_ratio)


def process_force_curve(x, fit_mode, k, s_ratio):
    rc, (zxr_and_dxr) = x
    npts = sum(map(len, zxr_and_dxr[0]))
    if npts > RESAMPLE_NPTS:
        zxr_and_dxr = np.reshape(zxr_and_dxr, (2, -1))
        zxr_and_dxr = resample_wrapper(zxr_and_dxr, RESAMPLE_NPTS, True)
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
    delta = z - d
    f = d * k
    return delta, f, split, rc
