import numpy as np
import warnings
import matplotlib.pyplot as plt


def isCPvector(tau, N, K):
    """
    Given tau an array of potential CP vectors, check whether these potential CP vectors
    are actually CP vectors as defined above
    """
    assert K + 1 <= N
    length = tau.shape[-1] == K + 2
    is_sorted = np.all(tau[..., :-1] < tau[..., 1:])
    init = (tau[..., 0] == -1).all()
    last = (tau[..., -1] == N - 1).all()
    return length and is_sorted and init and last


def eta(N, k, tau_det):
    """
    Return eta_k as defined in [§2.2, Article]
    """
    assert k >= 1
    indices = np.arange(N)
    return (1 / (tau_det[k] - tau_det[k - 1])) * (tau_det[k - 1] < indices) * (
        indices <= tau_det[k]
    ) - (1 / (tau_det[k + 1] - tau_det[k])) * (tau_det[k] < indices) * (
        indices <= tau_det[k + 1]
    )


def decompose(N, k, Sigma, tau_det, X):
    """
    Given a signal X, returns a,b as defined in [§3.1, Article]
    """
    eta_k = eta(N, k, tau_det)
    assert not np.isnan(eta_k).any()
    c = Sigma @ eta_k / (np.dot(eta_k, Sigma @ eta_k))
    return (X - np.dot(eta_k, X) * c), c


def get_QF_C(a, b, tau, kappa):
    """
    Compute the coefficient of the quadratic form z -> C(x(z)_{tau_{kappa-1}+1:tau_kappa})
    as used in [(11), Article]
    """
    # sum_i b_i**2 z**2 + 2za_ib_i + a_i**2 - (e - s + 1) * (mean(a)**2 + 2 * mean(a)mean(b)z + mean(b)**2 z**2)
    assert kappa >= 1
    assert kappa < len(tau)
    s = tau[kappa - 1] + 1
    e = tau[kappa]
    if e == s:
        return np.zeros(3)

    coeff = np.empty(3)
    cov_matrix = np.cov(np.stack([a[s : e + 1], b[s : e + 1]]))
    assert s < e
    assert not np.isnan(cov_matrix).any(), (s, e)
    coeff[0] = (e - s + 1) * cov_matrix[1, 1]  # np.var(b[s:e+1]) = dom coeff
    coeff[2] = (e - s + 1) * cov_matrix[0, 0]  # np.var(a[s:e+1])
    coeff[1] = 2 * (e - s + 1) * cov_matrix[0, 1]  # np.corrcoef(a[s:e+1], b[s:e+1])
    assert not np.isnan(coeff).any()
    return coeff


def get_QF(n, k, a, b, tau):
    return np.sum(
        np.stack([get_QF_C(a, b, tau, kappa) for kappa in range(1, k + 2)], axis=0),
        axis=0,
    )


def eval_poly(coeffs, z):
    """
    Given coeffs a matrix of shape (*, 3) where each row represents the coefficients of a quadratic form,
    eval those quadratic forms at point z
    """
    return coeffs[..., 0] * z ** 2 + coeffs[..., 1] * z + coeffs[..., 2]


def compute_break_points(a, b, c, z_u):
    """
    Compute the set of breakpoints appearing in [Lemma 1, Report]
    """
    Delta = b ** 2 - 4 * a * c

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = (-b - np.sqrt(Delta)) / (2 * a)

    degree_one = np.logical_and(np.isclose(a, 0), ~np.isclose(b, 0))
    sol[degree_one] = -c[degree_one] / b[degree_one]

    degree_zero = np.logical_and(np.isclose(a, 0), np.isclose(b, 0))
    sol[degree_zero] = float("nan")

    condition = np.logical_and(sol > z_u, np.logical_not(np.isnan(sol)))
    condition = np.logical_and(condition, Delta > 0)
    sol[np.logical_not(condition)] = float("inf")

    return sol


def next_break_point(a, b, c, z_u):
    """
    Perform one step of [Lemma 1, Report]
    """
    breakpoints = compute_break_points(a, b, c, z_u)

    break_ind = np.argmin(breakpoints, axis=None)
    break_val = np.min(breakpoints)

    if break_val == float("inf"):
        return None

    check_atol = 1e-8
    check_rtol = 1e-7
    if (
        not np.sum(
            np.isclose(
                breakpoints,
                break_val * np.ones_like(breakpoints),
                atol=check_atol,
                rtol=check_rtol,
            )
        )
        == 1
    ):
        print(breakpoints, break_val)
        indices = np.isclose(
            breakpoints,
            break_val * np.ones_like(breakpoints),
            atol=check_atol,
            rtol=check_rtol,
        ).nonzero()
        indices = indices[0]
        x = np.linspace(break_val - 1, break_val + 1)
        for ind in indices:
            plt.plot(x, a[ind] * x ** 2 + b[ind] * x + c[ind], label=str(ind))
        plt.legend()
        plt.show()

    assert (
        np.sum(
            np.isclose(
                breakpoints,
                break_val * np.ones_like(breakpoints),
                atol=check_atol,
                rtol=check_rtol,
            )
        )
        == 1
    )

    assert np.isclose(
        eval_poly(np.array([a[break_ind], b[break_ind], c[break_ind]]), break_val),
        0,
        atol=1e-8,
    ), (
        eval_poly(np.array([a[break_ind], b[break_ind], c[break_ind]]), break_val),
        np.array([a[break_ind], b[break_ind], c[break_ind]]),
        break_val,
    )
    return (break_ind, break_val)


def lex_min(arr):
    """
    Given arr a matrix, find the minimal row for the lexicographic order
    """
    min_row = arr[0]
    min_index = 0
    for i, row in enumerate(arr):
        j = 0
        while min_row[j] == row[j] and j < len(min_row) - 1:
            j += 1
        if min_row[j] > row[j]:
            min_row = row
            min_index = i
    return min_index, min_row


def is_lex_leq(arr, comp_row):
    """
    Given arr a matrix and comp_row a vector, test whether each row of arr
    is <= comp_row in lexicographic order
    """
    n, deg = arr.shape
    assert comp_row.shape == (deg,)

    is_leq = np.ones(n, dtype=bool)
    for i, row in enumerate(arr):
        j = 0
        while comp_row[j] == row[j] and j < deg - 1:
            j += 1
        assert comp_row[j] != row[j] or j == deg - 1
        is_leq[i] = row[j] <= comp_row[j]

    return is_leq


def invert_deg_one(arr, index=1):
    """ Reverse the second column of the matrix arr """
    new_arr = np.copy(arr)
    new_arr[..., index] = -new_arr[..., index]
    return new_arr