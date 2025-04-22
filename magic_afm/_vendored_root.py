"""
Spectral Algorithm for Nonlinear Equations
"""

# Vendored from scipy-1.15.2 under BSD license
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import collections

import numpy as np


class _NoConvergence(Exception):
    pass


def root_df_sane(
    func,
    x0,
    args=(),
    ftol=1e-8,
    fatol=1e-300,
    maxfev=1000,
    fnorm=None,
    callback=None,
    disp=False,
    M=10,
    eta_strategy=None,
    sigma_eps=1e-10,
    sigma_0=1.0,
    line_search="cruz",
):
    r"""
    Solve nonlinear equation with the DF-SANE method

    Options
    -------
    ftol : float, optional
        Relative norm tolerance.
    fatol : float, optional
        Absolute norm tolerance.
        Algorithm terminates when ``||func(x)|| < fatol + ftol ||func(x_0)||``.
    fnorm : callable, optional
        Norm to use in the convergence check. If None, 2-norm is used.
    maxfev : int, optional
        Maximum number of function evaluations.
    disp : bool, optional
        Whether to print convergence process to stdout.
    eta_strategy : callable, optional
        Choice of the ``eta_k`` parameter, which gives slack for growth
        of ``||F||**2``.  Called as ``eta_k = eta_strategy(k, x, F)`` with
        `k` the iteration number, `x` the current iterate and `F` the current
        residual. Should satisfy ``eta_k > 0`` and ``sum(eta, k=0..inf) < inf``.
        Default: ``||F||**2 / (1 + k)**2``.
    sigma_eps : float, optional
        The spectral coefficient is constrained to ``sigma_eps < sigma < 1/sigma_eps``.
        Default: 1e-10
    sigma_0 : float, optional
        Initial spectral coefficient.
        Default: 1.0
    M : int, optional
        Number of iterates to include in the nonmonotonic line search.
        Default: 10
    line_search : {'cruz', 'cheng'}
        Type of line search to employ. 'cruz' is the original one defined in
        [Martinez & Raydan. Math. Comp. 75, 1429 (2006)], 'cheng' is
        a modified search defined in [Cheng & Li. IMA J. Numer. Anal. 29, 814 (2009)].
        Default: 'cruz'

    References
    ----------
    .. [1] "Spectral residual method without gradient information for solving
           large-scale nonlinear systems of equations." W. La Cruz,
           J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).
    .. [2] W. La Cruz, Opt. Meth. Software, 29, 24 (2014).
    .. [3] W. Cheng, D.-H. Li. IMA J. Numer. Anal. **29**, 814 (2009).

    """

    if line_search not in ("cheng", "cruz"):
        raise ValueError(f"Invalid value {line_search!r} for 'line_search'")

    nexp = 2

    if eta_strategy is None:
        # Different choice from [1], as their eta is not invariant
        # vs. scaling of F.
        def eta_strategy(k, x, F):
            # Obtain squared 2-norm of the initial residual from the outer scope
            return f_0 / (1 + k) ** 2

    if fnorm is None:

        def fnorm(F):
            # Obtain squared 2-norm of the current residual from the outer scope
            return f_k ** (1.0 / nexp)

    def fmerit(F):
        return np.linalg.norm(F) ** nexp

    nfev = [0]
    f, x_k, x_shape, f_k, F_k, is_complex = _wrap_func(
        func, x0, fmerit, nfev, maxfev, args
    )

    k = 0
    f_0 = f_k
    sigma_k = sigma_0

    F_0_norm = fnorm(F_k)

    # For the 'cruz' line search
    prev_fs = collections.deque([f_k], M)

    # For the 'cheng' line search
    Q = 1.0
    C = f_0

    converged = False
    message = "too many function evaluations required"

    while True:
        F_k_norm = fnorm(F_k)

        if disp:
            print("iter %d: ||F|| = %g, sigma = %g" % (k, F_k_norm, sigma_k))

        if callback is not None:
            callback(x_k, F_k)

        if F_k_norm < ftol * F_0_norm + fatol:
            # Converged!
            message = "successful convergence"
            converged = True
            break

        # Control spectral parameter, from [2]
        if abs(sigma_k) > 1 / sigma_eps:
            sigma_k = 1 / sigma_eps * np.sign(sigma_k)
        elif abs(sigma_k) < sigma_eps:
            sigma_k = sigma_eps

        # Line search direction
        d = -sigma_k * F_k

        # Nonmonotone line search
        eta = eta_strategy(k, x_k, F_k)
        try:
            if line_search == "cruz":
                alpha, xp, fp, Fp = _nonmonotone_line_search_cruz(
                    f, x_k, d, prev_fs, eta=eta
                )
            elif line_search == "cheng":
                alpha, xp, fp, Fp, C, Q = _nonmonotone_line_search_cheng(
                    f, x_k, d, f_k, C, Q, eta=eta
                )
        except _NoConvergence:
            break

        # Update spectral parameter
        s_k = xp - x_k
        y_k = Fp - F_k
        sigma_k = np.vdot(s_k, s_k) / np.vdot(s_k, y_k)

        # Take step
        x_k = xp
        F_k = Fp
        f_k = fp

        # Store function value
        if line_search == "cruz":
            prev_fs.append(fp)

        k += 1

    x = _wrap_result(x_k, is_complex, shape=x_shape)

    return x


def _wrap_func(func, x0, fmerit, nfev_list, maxfev, args=()):
    """
    Wrap a function and an initial value so that (i) complex values
    are wrapped to reals, and (ii) value for a merit function
    fmerit(x, f) is computed at the same time, (iii) iteration count
    is maintained and an exception is raised if it is exceeded.

    Parameters
    ----------
    func : callable
        Function to wrap
    x0 : ndarray
        Initial value
    fmerit : callable
        Merit function fmerit(f) for computing merit value from residual.
    nfev_list : list
        List to store number of evaluations in. Should be [0] in the beginning.
    maxfev : int
        Maximum number of evaluations before _NoConvergence is raised.
    args : tuple
        Extra arguments to func

    Returns
    -------
    wrap_func : callable
        Wrapped function, to be called as
        ``F, fp = wrap_func(x0)``
    x0_wrap : ndarray of float
        Wrapped initial value; raveled to 1-D and complex
        values mapped to reals.
    x0_shape : tuple
        Shape of the initial value array
    f : float
        Merit function at F
    F : ndarray of float
        Residual at x0_wrap
    is_complex : bool
        Whether complex values were mapped to reals

    """
    x0 = np.asarray(x0)
    x0_shape = x0.shape
    F = np.asarray(func(x0, *args)).ravel()
    is_complex = np.iscomplexobj(x0) or np.iscomplexobj(F)
    x0 = x0.ravel()

    nfev_list[0] = 1

    if is_complex:

        def wrap_func(x):
            if nfev_list[0] >= maxfev:
                raise _NoConvergence()
            nfev_list[0] += 1
            z = _real2complex(x).reshape(x0_shape)
            v = np.asarray(func(z, *args)).ravel()
            F = _complex2real(v)
            f = fmerit(F)
            return f, F

        x0 = _complex2real(x0)
        F = _complex2real(F)
    else:

        def wrap_func(x):
            if nfev_list[0] >= maxfev:
                raise _NoConvergence()
            nfev_list[0] += 1
            x = x.reshape(x0_shape)
            F = np.asarray(func(x, *args)).ravel()
            f = fmerit(F)
            return f, F

    return wrap_func, x0, x0_shape, fmerit(F), F, is_complex


def _wrap_result(result, is_complex, shape=None):
    """
    Convert from real to complex and reshape result arrays.
    """
    if is_complex:
        z = _real2complex(result)
    else:
        z = result
    if shape is not None:
        z = z.reshape(shape)
    return z


def _real2complex(x):
    return np.ascontiguousarray(x, dtype=float).view(np.complex128)


def _complex2real(z):
    return np.ascontiguousarray(z, dtype=complex).view(np.float64)


# ------------------------------------------------------------------------------
# Non-monotone line search for DF-SANE
# ------------------------------------------------------------------------------


def _nonmonotone_line_search_cruz(
    f, x_k, d, prev_fs, eta, gamma=1e-4, tau_min=0.1, tau_max=0.5
):
    """
    Nonmonotone backtracking line search as described in [1]_

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    prev_fs : float
        List of previous merit function values. Should have ``len(prev_fs) <= M``
        where ``M`` is the nonmonotonicity window parameter.
    eta : float
        Allowed merit function increase, see [1]_
    gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position

    References
    ----------
    [1] "Spectral residual method without gradient information for solving
        large-scale nonlinear systems of equations." W. La Cruz,
        J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).

    """
    f_k = prev_fs[-1]
    f_bar = max(prev_fs)

    alpha_p = 1
    alpha_m = 1
    alpha = 1

    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)

        if fp <= f_bar + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        alpha_tp = alpha_p**2 * f_k / (fp + (2 * alpha_p - 1) * f_k)

        xp = x_k - alpha_m * d
        fp, Fp = f(xp)

        if fp <= f_bar + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        alpha_tm = alpha_m**2 * f_k / (fp + (2 * alpha_m - 1) * f_k)

        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    return alpha, xp, fp, Fp


def _nonmonotone_line_search_cheng(
    f, x_k, d, f_k, C, Q, eta, gamma=1e-4, tau_min=0.1, tau_max=0.5, nu=0.85
):
    """
    Nonmonotone line search from [1]

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    f_k : float
        Initial merit function value.
    C, Q : float
        Control parameters. On the first iteration, give values
        Q=1.0, C=f_k
    eta : float
        Allowed merit function increase, see [1]_
    nu, gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position
    C : float
        New value for the control parameter C
    Q : float
        New value for the control parameter Q

    References
    ----------
    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line
           search and its application to the spectral residual
           method'', IMA J. Numer. Anal. 29, 814 (2009).

    """
    alpha_p = 1
    alpha_m = 1
    alpha = 1

    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)

        if fp <= C + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        alpha_tp = alpha_p**2 * f_k / (fp + (2 * alpha_p - 1) * f_k)

        xp = x_k - alpha_m * d
        fp, Fp = f(xp)

        if fp <= C + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        alpha_tm = alpha_m**2 * f_k / (fp + (2 * alpha_m - 1) * f_k)

        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    # Update C and Q
    Q_next = nu * Q + 1
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next

    return alpha, xp, fp, Fp, C, Q
