# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
This module implements a Levenberg-Marquardt algorithm with constraints on the
fitted parameters without introducing any other dependendency than numpy.

If scipy dependency is not an issue, and no constraints are applied to the fitting
parameters, there is no real gain compared to the use of scipy.optimize.curve_fit
other than a more conservative calculation of uncertainties on fitted parameters.

This module is a refactored version of PyMca Gefit.py module.
"""
__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "15/05/2017"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import copy

import numpy

# codes understood by the routine
CFREE = 0
CPOSITIVE = 1
CQUOTED = 2
CFIXED = 3
CFACTOR = 4
CDELTA = 5
CSUM = 6
CIGNORED = 7


def leastsq(
    model,
    xdata,
    ydata,
    p0,
    sigma=None,
    constraints=None,
    model_deriv=None,
    epsfcn=None,
    deltachi=None,
    full_output=None,
    left_derivative=False,
    max_iter=100,
):
    """
    Use non-linear least squares Levenberg-Marquardt algorithm to fit a function, f, to
    data with optional constraints on the fitted parameters.

    Assumes ``ydata = f(xdata, *params) + eps``

    :param model: callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
        The returned value is a one dimensional array of floats.

    :param xdata: An M-length sequence.
        The independent variable where the data is measured.

    :param ydata: An M-length sequence
        The dependent data --- nominally f(xdata, ...)

    :param p0: N-length sequence
        Initial guess for the parameters.

    :param sigma: None or M-length sequence, optional
        If not None, the uncertainties in the ydata array. These are used as
        weights in the least-squares problem
        i.e. minimising ``np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )``
        If None, the uncertainties are assumed to be 1

    :param constraints:
        If provided, it is a 2D sequence of dimension (n_parameters, 3) where,
        for each parameter denoted by the index i, the meaning is

                     - constraints[i][0]

                        - 0 - Free (CFREE)
                        - 1 - Positive (CPOSITIVE)
                        - 2 - Quoted (CQUOTED)
                        - 3 - Fixed (CFIXED)
                        - 4 - Factor (CFACTOR)
                        - 5 - Delta (CDELTA)
                        - 6 - Sum (CSUM)


                     - constraints[i][1]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Min value of the parameter if constraints[i][0] is CQUOTED
                        - Index of fitted parameter to which it is related

                     - constraints[i][2]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Max value of the parameter if constraints[i][0] is CQUOTED
                        - Factor to apply to related parameter with index constraints[i][1]
                        - Difference with parameter with index constraints[i][1]
                        - Sum obtained when adding parameter with index constraints[i][1]
    :type constraints: *optional*, None or 2D sequence

    :param model_deriv:
        None (default) or function providing the derivatives of the fitting function respect to the fitted parameters.
        It will be called as model_deriv(xdata, parameters, index) where parameters is a sequence with the current
        values of the fitting parameters, index is the fitting parameter index for which the the derivative has
        to be provided in the supplied array of xdata points.
    :type model_deriv: *optional*, None or callable


    :param epsfcn: float
        A variable used in determining a suitable parameter variation when
        calculating the numerical derivatives (for model_deriv=None).
        Normally the actual step length will be sqrt(epsfcn)*x
        Original Gefit module was using epsfcn 1.0e-5 while default value
        is now numpy.finfo(numpy.float32).eps
    :type epsfcn: *optional*, float

    :param deltachi: float
        A variable used to control the minimum change in chisq to consider the
        fitting process not worth to be continued. Default is 0.1 %.
    :type deltachi: *optional*, float

    :param full_output: bool, optional
        non-zero to return all optional outputs. The default is None what will give a warning in case
        of a constrained fit without having set this kweyword.
    :param left_derivative:
            This parameter only has an influence if no derivative function
            is provided. When True the left and right derivatives of the
            model will be calculated for each fitted parameters thus leading to
            the double number of function evaluations. Default is False.
            Original Gefit module was always using left_derivative as True.
    :type left_derivative: *optional*, bool

    :param max_iter: Maximum number of iterations (default is 100)

    :return: Returns a tuple of length 2 (or 3 if full_ouput is True) with the content:

         ``popt``: array
           Optimal values for the parameters so that the sum of the squared error
           of ``f(xdata, *popt) - ydata`` is minimized
         ``pcov``: 2d array
           If no constraints are applied, this array contains the estimated covariance
           of popt. The diagonal provides the variance of the parameter estimate.
           To compute one standard deviation errors use ``perr = np.sqrt(np.diag(pcov))``.
           If constraints are applied, this array does not contain the estimated covariance of
           the parameters actually used during the fitting process but the uncertainties after
           recalculating the covariance if all the parameters were free.
           To get the actual uncertainties following error propagation of the actually fitted
           parameters one should set full_output to True and access the uncertainties key.
         ``infodict``: dict
           a dictionary of optional outputs with the keys:

            ``uncertainties``
                The actual uncertainty on the optimized parameters.
            ``nfev``
                The number of function calls
            ``fvec``
                The function evaluated at the output
            ``niter``
                The number of iterations performed
            ``chisq``
                The chi square ``np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )``
            ``reduced_chisq``
                The chi square ``np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )`` divided
                by the number of degrees of freedom ``(M - number_of_free_parameters)``
    """
    parameters = numpy.asarray(p0, dtype=float)
    if deltachi is None:
        deltachi = 0.001

    # NaNs can not be handled
    xdata = numpy.asarray_chkfinite(xdata)
    ydata = numpy.asarray_chkfinite(ydata)
    if sigma is not None:
        sigma = numpy.asarray_chkfinite(sigma)
    else:
        sigma = numpy.ones(ydata.shape, dtype=float)
    ydata.shape = -1
    sigma.shape = -1

    weight = 1.0 / (sigma + numpy.equal(sigma, 0))
    weight0 = weight * weight

    nparameters = len(parameters)

    if epsfcn is None:
        epsfcn = numpy.finfo(numpy.float32).eps
    else:
        epsfcn = max(epsfcn, numpy.finfo(numpy.float32).eps)

    if constraints is not None:
        constraints = numpy.array(constraints)

    # Levenberg-Marquardt algorithm
    fittedpar = parameters.__copy__()
    flambda = 0.001
    iiter = max_iter
    last_evaluation = None
    x = xdata
    y = ydata
    chisq0 = -1
    iteration_counter = 0
    function_call_counter = 0
    while iiter > 0:
        weight = weight0
        """
        I cannot evaluate the initial chisq here because I do not know
        if some parameters are to be ignored, otherways I could do it as follows:
        if last_evaluation is None:
            yfit = model(x, *fittedpar)
            last_evaluation = yfit
            chisq0 = (weight * pow(y-yfit, 2)).sum()
        and chisq would not need to be recalculated.
        Passing the last_evaluation assumes that there are no parameters being
        ignored or not between calls.
        """
        iteration_counter += 1
        chisq0, alpha0, beta, internal_output = chisq_alpha_beta(
            model,
            fittedpar,
            x,
            y,
            weight,
            constraints=constraints,
            model_deriv=model_deriv,
            epsfcn=epsfcn,
            left_derivative=left_derivative,
            last_evaluation=last_evaluation,
            full_output=True,
        )
        n_free = internal_output["n_free"]
        free = internal_output["free"]
        noigno = internal_output["noigno"]
        fitparam = internal_output["fitparam"]
        function_calls = internal_output["function_calls"]
        function_call_counter += function_calls
        nr, nc = alpha0.shape
        flag = 0
        while flag == 0:
            alpha = alpha0 * (1.0 + flambda * numpy.identity(nr))
            deltapar = numpy.linalg.solve(alpha.T, beta.T).T @ numpy.diag(free)[free]
            deltapar = deltapar[0]
            newpar = fitparam + deltapar
            if constraints is not None:
                for i, (cons, cmin, cmax) in enumerate(constraints):
                    if cons == CQUOTED:
                        pmax = max(cmin, cmax)
                        pmin = min(cmin, cmax)
                        A = 0.5 * (pmax + pmin)
                        B = 0.5 * (pmax - pmin)
                        if B != 0:
                            newpar[i] = A + B * numpy.sin(
                                numpy.arcsin((fitparam[i] - A) / B) + deltapar[i]
                            )
                        else:
                            txt = "Error processing constrained fit\n"
                            txt += "Parameter limits are %g and %g\n" % (pmin, pmax)
                            txt += "A = %g B = %g" % (A, B)
                            raise ValueError("Invalid parameter limits")
                newpar = _get_parameters(newpar, constraints)
            workpar = newpar[noigno]
            yfit = model(x, *workpar)
            yfit.shape = -1
            function_call_counter += 1
            chisq = (weight * (y - yfit) ** 2).sum()
            absdeltachi = chisq0 - chisq
            if absdeltachi < 0:
                flambda *= 10.0
                if flambda > 1000:
                    flag = 1
                    iiter = 0
                    last_evaluation = yfit
            else:
                flag = 1
                fittedpar = newpar.__copy__()
                lastdeltachi = 100 * (absdeltachi / (chisq + (chisq == 0)))
                if iteration_counter < 2:
                    # ignore any limit, the fit *has* to be improved
                    pass
                elif (lastdeltachi) < deltachi:
                    iiter = 0
                elif absdeltachi < numpy.sqrt(epsfcn):
                    iiter = 0
                chisq0 = chisq
                flambda = flambda / 10.0
                last_evaluation = yfit
            iiter = iiter - 1
    # this is the covariance matrix of the actually fitted parameters
    cov0 = numpy.linalg.pinv(alpha0)
    if constraints is None:
        cov = cov0
    else:
        # yet another call needed with all the parameters being free except those
        # that are FIXED and that will be assigned a 100 % uncertainty.
        new_constraints = copy.deepcopy(constraints)
        flag_special = [0] * len(fittedpar)
        for idx, constraint in enumerate(constraints):
            if constraints[idx][0] in [CFIXED, CIGNORED]:
                flag_special[idx] = constraints[idx][0]
            else:
                new_constraints[idx][0] = CFREE
                new_constraints[idx][1] = 0
                new_constraints[idx][2] = 0
        chisq, alpha, beta, internal_output = chisq_alpha_beta(
            model,
            fittedpar,
            x,
            y,
            weight,
            constraints=new_constraints,
            model_deriv=model_deriv,
            epsfcn=epsfcn,
            left_derivative=left_derivative,
            last_evaluation=last_evaluation,
            full_output=True,
        )
        cov = numpy.linalg.pinv(alpha)
        for idx, value in enumerate(flag_special):
            if value in [CFIXED, CIGNORED]:
                cov = numpy.insert(numpy.insert(cov, idx, 0, axis=1), idx, 0, axis=0)
                cov[idx, idx] = fittedpar[idx] * fittedpar[idx]

    if not full_output:
        return fittedpar, cov
    else:
        sigma0 = numpy.sqrt(abs(numpy.diag(cov0)))
        sigmapar = _get_sigma_parameters(fittedpar, sigma0, constraints)
        ddict = {}
        ddict["chisq"] = chisq0
        ddict["reduced_chisq"] = chisq0 / (len(yfit) - n_free)
        ddict["covariance"] = cov0
        ddict["uncertainties"] = sigmapar
        ddict["fvec"] = last_evaluation
        ddict["nfev"] = function_call_counter
        ddict["niter"] = iteration_counter
        return fittedpar, cov, ddict


def chisq_alpha_beta(
    model,
    parameters,
    x,
    y,
    weight,
    constraints=None,
    model_deriv=None,
    epsfcn=None,
    left_derivative=False,
    last_evaluation=None,
    full_output=False,
):
    """
    Get chi square, the curvature matrix alpha and the matrix beta according to the input parameters.
    If all the parameters are unconstrained, the covariance matrix is the inverse of the alpha matrix.

    :param model: callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
        The returned value is a one dimensional array of floats.

    :param parameters: N-length sequence
        Values of parameters at which function and derivatives are to be calculated.

    :param x: An M-length sequence.
        The independent variable where the data is measured.

    :param y: An M-length sequence
        The dependent data --- nominally f(xdata, ...)

    :param weight: M-length sequence
        Weights to be applied in the calculation of chi square
        As a reminder ``chisq = np.sum(weigth * (model(x, *parameters) - y)**2)``

    :param constraints:
        If provided, it is a 2D sequence of dimension (n_parameters, 3) where,
        for each parameter denoted by the index i, the meaning is

                     - constraints[i][0]

                        - 0 - Free (CFREE)
                        - 1 - Positive (CPOSITIVE)
                        - 2 - Quoted (CQUOTED)
                        - 3 - Fixed (CFIXED)
                        - 4 - Factor (CFACTOR)
                        - 5 - Delta (CDELTA)
                        - 6 - Sum (CSUM)


                     - constraints[i][1]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Min value of the parameter if constraints[i][0] is CQUOTED
                        - Index of fitted parameter to which it is related

                     - constraints[i][2]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Max value of the parameter if constraints[i][0] is CQUOTED
                        - Factor to apply to related parameter with index constraints[i][1]
                        - Difference with parameter with index constraints[i][1]
                        - Sum obtained when adding parameter with index constraints[i][1]
    :type constraints: *optional*, None or 2D sequence

    :param model_deriv:
        None (default) or function providing the derivatives of the fitting function respect to the fitted parameters.
        It will be called as model_deriv(xdata, parameters, index) where parameters is a sequence with the current
        values of the fitting parameters, index is the fitting parameter index for which the the derivative has
        to be provided in the supplied array of xdata points.
    :type model_deriv: *optional*, None or callable


    :param epsfcn: float
        A variable used in determining a suitable parameter variation when
        calculating the numerical derivatives (for model_deriv=None).
        Normally the actual step length will be sqrt(epsfcn)*x
        Original Gefit module was using epsfcn 1.0e-10 while default value
        is now numpy.finfo(numpy.float32).eps
    :type epsfcn: *optional*, float

    :param left_derivative:
            This parameter only has an influence if no derivative function
            is provided. When True the left and right derivatives of the
            model will be calculated for each fitted parameters thus leading to
            the double number of function evaluations. Default is False.
            Original Gefit module was always using left_derivative as True.
    :type left_derivative: *optional*, bool

    :param last_evaluation: An M-length array
            Used for optimization purposes. If supplied, this array will be taken as the result of
            evaluating the function, that is as the result of ``model(x, *parameters)`` thus avoiding
            the evaluation call.

    :param full_output: bool, optional
            Additional output used for internal purposes with the keys:
        ``function_calls``
            The number of model function calls performed.
        ``fitparam``
            A sequence with the actual free parameters
        ``free``
            Sequence with the indices of the free parameters in input parameters sequence.
        ``noigno``
            Sequence with the indices of the original parameters considered in the calculations.
    """
    if epsfcn is None:
        epsfcn = numpy.finfo(numpy.float32).eps
    else:
        epsfcn = max(epsfcn, numpy.finfo(numpy.float32).eps)

    n_param = len(parameters)
    derivfactor = numpy.ones_like(parameters, dtype=numpy.float64)
    noigno = numpy.ones_like(parameters, dtype=numpy.bool)
    free = numpy.zeros_like(parameters, dtype=numpy.bool)
    fitparam = numpy.copy(parameters)
    n_free = 0

    if constraints is not None:
        for i in range(n_param):
            if constraints[i][0] == CIGNORED:
                noigno[i] = False
            if constraints[i][0] == CFREE:
                fitparam[i] = parameters[i]
                free[i] = True
                n_free += 1
            elif constraints[i][0] == CPOSITIVE:

                fitparam[i] = abs(parameters[i])
                free[i] = True
                n_free += 1
            elif constraints[i][0] == CQUOTED:
                pmax = max(constraints[i][1], constraints[i][2])
                pmin = min(constraints[i][1], constraints[i][2])
                if (
                    ((pmax - pmin) > 0)
                    & (parameters[i] <= pmax)
                    & (parameters[i] >= pmin)
                ):
                    A = 0.5 * (pmax + pmin)
                    B = 0.5 * (pmax - pmin)
                    fitparam[i] = parameters[i]
                    derivfactor[i] *= B * numpy.cos(
                        numpy.arcsin((parameters[i] - A) / B)
                    )
                    free[i] = True
                    n_free += 1
                else:
                    raise ValueError("Constraint violation", pmin, parameters[i], pmax)

    delta = (fitparam + numpy.equal(fitparam, 0.0)) * numpy.sqrt(epsfcn)
    nr = y.size
    ##############
    # Prior to each call to the function one has to re-calculate the
    # parameters
    pwork = parameters.__copy__()
    pwork[free] = fitparam[free]
    if n_free == 0:
        raise ValueError("No free parameters to fit")
    function_calls = 0
    if not left_derivative:
        if last_evaluation is not None:
            f2 = last_evaluation
        else:
            f2 = model(x, *parameters)
            f2.shape = -1
            function_calls += 1
    deriv = numpy.zeros((n_free, nr))
    for deriv_inx, i in enumerate(numpy.where(free)[0]):
        if model_deriv is None:
            pwork[i] = fitparam[i] + delta[i]
            newpar = _get_parameters(pwork, constraints)
            newpar = newpar[noigno]
            f1 = model(x, *newpar)
            function_calls += 1
            if left_derivative:
                pwork[i] = fitparam[i] - delta[i]
                newpar = _get_parameters(pwork, constraints)
                newpar = newpar[noigno]
                f2 = model(x, *newpar)
                function_calls += 1
                help0 = (f1 - f2) / (2.0 * delta[i])
            else:
                help0 = (f1 - f2) / (delta[i])
            deriv[deriv_inx] = help0 * derivfactor[i]
            pwork[i] = fitparam[i]
        else:
            help0 = model_deriv(x, pwork, i)
            deriv[deriv_inx] = help0 * derivfactor[i]

    if last_evaluation is None:
        if constraints is None:
            yfit = model(x, *fitparam)
            yfit.shape = -1
        else:
            newpar = _get_parameters(pwork, constraints)
            newpar = newpar[noigno]
            yfit = model(x, *newpar)
            yfit.shape = -1
        function_calls += 1
    else:
        yfit = last_evaluation
    deltay = y - yfit
    help0 = weight * deltay
    alpha = deriv @ (weight * deriv).T
    beta = help0[numpy.newaxis, :] @ deriv.T
    chisq = (help0 * deltay).sum()
    if full_output:
        ddict = {}
        ddict["n_free"] = n_free
        ddict["free"] = free
        ddict["noigno"] = noigno
        ddict["fitparam"] = fitparam
        ddict["derivfactor"] = derivfactor
        ddict["function_calls"] = function_calls
        return chisq, alpha, beta, ddict
    else:
        return chisq, alpha, beta


def _get_parameters(parameters, constraints):
    """
    Apply constraints to input parameters.

    Parameters not depending on other parameters, they are returned as the input.

    Parameters depending on other parameters, return the value after applying the
    relation to the parameter wo which they are related.
    """
    # 0 = Free       1 = Positive     2 = Quoted
    # 3 = Fixed      4 = Factor       5 = Delta
    if constraints is None:
        return numpy.copy(parameters)
    newparam = numpy.zeros_like(parameters)
    # first I make the free parameters
    # because the quoted ones put troubles
    for i in range(len(constraints)):
        if constraints[i][0] == CFREE:
            newparam[i] = parameters[i]
        elif constraints[i][0] == CPOSITIVE:
            newparam[i] = abs(parameters[i])
        elif constraints[i][0] == CQUOTED:
            newparam[i] = parameters[i]
        elif abs(constraints[i][0]) == CFIXED:
            newparam[i] = parameters[i]
        else:
            newparam[i] = parameters[i]
    for i in range(len(constraints)):
        if constraints[i][0] == CFACTOR:
            newparam[i] = constraints[i][2] * newparam[int(constraints[i][1])]
        elif constraints[i][0] == CDELTA:
            newparam[i] = constraints[i][2] + newparam[int(constraints[i][1])]
        elif constraints[i][0] == CIGNORED:
            # The whole ignored stuff should not be documented because setting
            # a parameter to 0 is not the same as being ignored.
            # Being ignored should imply the parameter is simply not accounted for
            # and should be stripped out of the list of parameters by the program
            # using this module
            newparam[i] = 0
        elif constraints[i][0] == CSUM:
            newparam[i] = constraints[i][2] - newparam[int(constraints[i][1])]
    return newparam


def _get_sigma_parameters(parameters, sigma0, constraints):
    """
    Internal function propagating the uncertainty on the actually fitted
    parameters and related parameters to the
    final parameters considering the applied constraints.

    Parameters
    ----------
        parameters : 1D sequence of length equal to the number of free parameters N
            The parameters actually used in the fitting process.
        sigma0 : 1D sequence of length N
            Uncertainties calculated as the square-root of the diagonal of
            the covariance matrix
        constraints : The set of constraints applied in the fitting process
    """
    # 0 = Free       1 = Positive     2 = Quoted
    # 3 = Fixed      4 = Factor       5 = Delta
    if constraints is None:
        return sigma0
    n_free = 0
    sigma_par = numpy.zeros_like(parameters, dtype=numpy.float64)
    for i in range(len(constraints)):
        if constraints[i][0] == CFREE:
            sigma_par[i] = sigma0[n_free]
            n_free += 1
        elif constraints[i][0] == CPOSITIVE:
            sigma_par[i] = sigma0[n_free]
            n_free += 1
        elif constraints[i][0] == CQUOTED:
            pmax = max(constraints[i][1], constraints[i][2])
            pmin = min(constraints[i][1], constraints[i][2])
            B = 0.5 * (pmax - pmin)
            if (B > 0) & (parameters[i] < pmax) & (parameters[i] > pmin):
                sigma_par[i] = abs(B * numpy.cos(parameters[i]) * sigma0[n_free])
                n_free += 1
            else:
                sigma_par[i] = parameters[i]
        elif abs(constraints[i][0]) == CFIXED:
            sigma_par[i] = parameters[i]
    for i in range(len(constraints)):
        if constraints[i][0] == CFACTOR:
            sigma_par[i] = constraints[i][2] * sigma_par[int(constraints[i][1])]
        elif constraints[i][0] == CDELTA:
            sigma_par[i] = sigma_par[int(constraints[i][1])]
        elif constraints[i][0] == CSUM:
            sigma_par[i] = sigma_par[int(constraints[i][1])]
    return sigma_par
