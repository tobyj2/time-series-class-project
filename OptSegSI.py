import numpy as np
import utilities as utl
from scipy.stats import norm
from scipy.special import logsumexp
from paraDP import paraDP


def add_interval_proba(log_proba, a, b, logcdf):
    """
    Return sum of probability specified by log_proba, and cdf(b)-cdf(a) where cdf is the cumulative distribution function corresponding to logcdf
    The distribution of the law specified by logcdf is assumed to be symmetric with respect to 0.
    """
    assert b > a
    if a > 0:
        a, b = -b, -a
    lb = logcdf(b)
    la = logcdf(a)
    return logsumexp(
        np.array([logsumexp(np.array([la, lb]), b=np.array([-1, 1])), log_proba])
    )


def log_truncated_cdf(x, intervals, var, verbose=False):
    """
    Return the log of the cdf of the N(0, var) conditioned on being in intervals
    """
    logcdf = norm(loc=0, scale=np.sqrt(var)).logcdf
    logdenom = -float("inf")
    lognum = -float("inf")
    assert len(intervals) > 0
    myprint = print if verbose else lambda x: ()
    myprint(f"x={x}, intervals={intervals}")
    for (a, b) in intervals:
        assert a < b
        if b < x:
            newlognum = add_interval_proba(lognum, a, b, logcdf)
            myprint(
                f"Case b<x: lognum = {newlognum}, a={a}, b= {b} (logcdf(b)= {logcdf(b)}, logcdf(a)={logcdf(a)})"
            )
            assert newlognum >= lognum
            lognum = newlognum
        elif a < x:
            newlognum = add_interval_proba(lognum, a, x, logcdf)
            myprint(
                f"Case a<x: lognum= {newlognum}, a={a}, x= {x} (logcdf(x)= {logcdf(x)}, logcdf(a)={logcdf(a)})"
            )
            assert newlognum >= lognum
            lognum = newlognum

        newlogdenom = add_interval_proba(logdenom, a, b, logcdf)
        assert newlogdenom >= logdenom
        logdenom = newlogdenom

        assert logdenom >= lognum

        myprint(
            f"logdemon = {logdenom}, a={a}, b= {b} (logcdf(b)= {logcdf(b)}, logcdf(a)={logcdf(a)})"
        )
        assert logdenom != -float("inf"), logdenom
    myprint(f"logdenom ={logdenom}, lognum = {lognum}, var={var}")

    return lognum - logdenom


def selective_p_value(z, intervals, var):
    """
    Compute the selective p-value when z ~ N(0, var) and conditioned on z in intervals
    """
    cdf_neg_z = log_truncated_cdf(-np.abs(z), intervals, var)
    cdf_pos_z = log_truncated_cdf(np.abs(z), intervals, var)
    assert cdf_neg_z <= cdf_pos_z
    return logsumexp(
        np.array([cdf_neg_z, logsumexp(np.array([0, cdf_pos_z]), b=np.array([1, -1]))])
    )


def get_tau_det(x_obs, N, K, Sigma, verbose=False):
    """
    Solve the standard CP detection problem using our algorithms
    (overkill)
    """
    k = 1
    tau_init = np.concatenate((np.arange(-1, K), np.array([N - 1])))
    eta_k = utl.eta(N, k, tau_init)
    a, b = utl.decompose(N, k, Sigma, tau_init, x_obs)
    z = np.dot(eta_k, x_obs)
    if verbose:
        print(f"a={a}, b={b}, z={z}")
    assert np.allclose(a + z * b, x_obs)
    lst_z, lst_t = paraDP(N, K, a, b, plot=False)
    lst_z += [float("inf")]
    for beg, end, t in zip(lst_z[:-1], lst_z[1:], lst_t):
        if beg <= z and z < end:
            return t


def optSegSI(
    x_obs,
    N,
    K,
    Sigma,
    verbose=False,
    with_tau_det=False,
    pruning=True,
    recalculate_Sigma=False,
):
    """
    Main function: compute the selective p-values at each detected CP
    """
    p_values = np.empty(K)
    tau_det = get_tau_det(x_obs, N, K, Sigma)

    if recalculate_Sigma:
        Sigma = np.max(
            [
                x_obs[tau_det[i] + 1 : tau_det[i + 1]].var()
                for i in range(len(tau_det) - 1)
            ]
        ) * np.identity(N)
    if verbose:
        print(f"tau_det = {tau_det}")
    for k, tau_k in enumerate(tau_det[1:-1], 1):
        eta_k = utl.eta(N, k, tau_det)
        a, b = utl.decompose(N, k, Sigma, tau_det, x_obs)
        z = np.dot(eta_k, x_obs)
        if verbose:
            print(f"a={a}, b={b}, z={z}")
        assert np.allclose(a + z * b, x_obs)
        lst_z, lst_t = paraDP(N, K, a, b, plot=False, pruning=pruning)
        lst_z += [float("inf")]
        intervals = []
        for beg, end, t in zip(lst_z[:-1], lst_z[1:], lst_t):
            if beg <= z and z < end:
                assert (
                    t == tau_det
                ).all(), (
                    f"t = {t}, tau_det = {tau_det},lst_z = {lst_z}, lst_t = {lst_t}  "
                )
            if (t == tau_det).all():
                intervals.append((beg, end))
        p_values[k - 1] = selective_p_value(z, intervals, np.dot(eta_k, Sigma @ eta_k))
    if not with_tau_det:
        return p_values
    else:
        return p_values, tau_det