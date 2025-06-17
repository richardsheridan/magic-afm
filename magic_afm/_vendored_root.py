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

import numpy as np



maxfev = 1000

def root_df_sane(
    func,
    x0,
    args=(),
    ftol=1e-8,
    fatol=1e-300,
    callback=None,
    disp=False,
    eta_strategy=None,
    sigma_eps=1e-10,
    sigma_0=1.0,
):
    r"""
    Solve nonlinear equation with the DF-SANE method w/Cheng linesearch

    Options
    -------
    ftol : float, optional
        Relative norm tolerance.
    fatol : float, optional
        Absolute norm tolerance.
        Algorithm terminates when ``||func(x)|| < fatol + ftol ||func(x_0)||``.
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

    References
    ----------
    .. [1] "Spectral residual method without gradient information for solving
           large-scale nonlinear systems of equations." W. La Cruz,
           J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).
    .. [2] W. La Cruz, Opt. Meth. Software, 29, 24 (2014).
    .. [3] W. Cheng, D.-H. Li. IMA J. Numer. Anal. **29**, 814 (2009).

    """

    if eta_strategy is None:
        # Different choice from [1], as their eta is not invariant
        # vs. scaling of F.
        def eta_strategy(k, x, F):
            # Obtain squared 2-norm of the initial residual from the outer scope
            return f_0 / (1 + k) ** 2

    x_k = x0
    F_k = func(x0, *args)
    f_k = np.sum(F_k * F_k)

    nfev = 1  # starts from one because of above call to func

    k = 0
    f_0 = f_k
    sigma_k = sigma_0

    F_0_norm = np.sqrt(f_k)

    # For the 'cheng' line search
    Q = 1.0
    C = f_0

    converged = False

    while True:
        F_k_norm = np.sqrt(f_k)

        if disp:
            print(f"iter {k}: nfev = {nfev}, ||F|| = {F_k_norm:g}, sigma = {sigma_k:g}")

        if callback is not None:
            callback(x_k, F_k)

        if F_k_norm < ftol * F_0_norm + fatol:
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

        search_res = _nonmonotone_line_search_cheng(
            func, args, x_k, d, f_k, C, Q, eta, nfev
        )
        if search_res is None:
            break
        alpha, xp, fp, Fp, C, Q, nfev = search_res

        # Update spectral parameter
        s_k = xp - x_k
        y_k = Fp - F_k
        sigma_k = np.sum(s_k * s_k) / np.sum(s_k * y_k)

        # Take step
        x_k = xp
        F_k = Fp
        f_k = fp

        k += 1

    return x_k


# ------------------------------------------------------------------------------
# Non-monotone line search for DF-SANE
# ------------------------------------------------------------------------------

def _nonmonotone_line_search_cheng(
    func,
    args,
    x_k,
    d,
    f_k,
    C,
    Q,
    eta,
    nfev,
    gamma=1e-4,
    tau_min=0.1,
    tau_max=0.5,
    nu=0.85,
):
    """
    Nonmonotone line search from [1]

    Parameters
    ----------
    f : callable
        Function returning ``F`` the residual.
    args : tuple
        Arguments for f.
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
    nfev : int
        function evaluation state passthrough
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
    alpha_p = np.array((1.0,))
    alpha_m = np.array((1.0,))
    alpha = np.array((1.0,))

    while True:
        xp = x_k + alpha_p * d
        Fp = func(xp, *args)
        nfev += 1
        if nfev >= maxfev:
            return None
        fp = np.sum(Fp * Fp)

        if fp <= C + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        alpha_tp = alpha_p**2 * f_k / (fp + (2.0 * alpha_p - 1.0) * f_k)

        xp = x_k - alpha_m * d
        Fp = func(xp, *args)
        nfev += 1
        if nfev >= maxfev:
            return None
        fp = np.sum(Fp * Fp)

        if fp <= C + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        alpha_tm = alpha_m**2 * f_k / (fp + (2.0 * alpha_m - 1.0) * f_k)

        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    # Update C and Q
    Q_next = nu * Q + 1.0
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next

    return alpha, xp, fp, Fp, C, Q, nfev
