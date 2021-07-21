import numpy as np
import matplotlib.pyplot as plt
import utilities as utl
from paraCP import paraCP


def pruning_step(n, k, a, b, former_T_hat, T_opt, lst_z_opt):
    """
    Compute the rightmost set of the equation above
    former_T_hat: corresponds to \bar{\mathcal T}_{k, n-1} above
    T_opt, lst_z_opt: describe L^{opt}_{k-1, n-1}
    """

    assert utl.isCPvector(former_T_hat, n - 1, k)
    assert utl.isCPvector(T_opt, n - 1, k - 1)
    assert len(lst_z_opt) == len(T_opt)

    lst_z = lst_z_opt + [float("inf")]

    qfs = np.stack([utl.get_QF(n, k, a, b, tau) for tau in former_T_hat], axis=0)

    # Init at -infty
    qf_opt = utl.get_QF(n, k - 1, a, b, T_opt[0])
    assert lst_z[0] == -float("inf")
    # These qfs satisfy the 'non-pruning' condition at - infty
    keep = utl.is_lex_leq(utl.invert_deg_one(qfs), utl.invert_deg_one(qf_opt))

    for beg, end, t in zip(lst_z[:-1], lst_z[1:], T_opt):
        # Induction hypothesis:
        # the qfs in ~keep were strictly above L^{opt}_{k-1, n-1}, beg included
        qf_opt = utl.get_QF(n, k - 1, a, b, t)
        differences = qfs[~keep] - qf_opt[None, :]
        bks = utl.compute_break_points(*(differences.T), beg)
        new_to_keep = (bks <= end) if end != float("inf") else (bks != float("inf"))
        keep[~keep] = new_to_keep

    return former_T_hat[keep]


def incr_length(tau_arr, new_n):
    """
    Given tau_arr an array of CP vectors for sequences of length < new_n,
    update them for sequences of length new_n
    """
    assert (tau_arr < new_n - 1).all()
    tau_arr[..., -1] = new_n - 1


def full_pruning(n, k, a, b, former_T_hat, T_opt, lst_z_opt):
    """
    Compute the full \bar{\mathcal T}_{k, n} as describe above
    """
    not_pruned = pruning_step(n, k, a, b, former_T_hat, T_opt, lst_z_opt)
    incr_length(not_pruned, n)
    assert utl.isCPvector(not_pruned, n, k)

    T_bar = np.concatenate((expand(T_opt, n), not_pruned), axis=0)
    assert utl.isCPvector(T_bar, n, k)

    return T_bar


def expand(matrix_cp_vectors, n):
    """
    Given matrix_cp_vectors an array of CP vectors for sequences of length m < n,
    add the CP m-1 and update them for sequences of length n
    """
    copy_matrix_cp_vectors = np.copy(matrix_cp_vectors)
    res = np.concatenate(
        (
            copy_matrix_cp_vectors,
            (n - 1) * np.ones(copy_matrix_cp_vectors.shape[0])[:, None],
        ),
        axis=1,
    )
    assert res.shape == (
        copy_matrix_cp_vectors.shape[0],
        copy_matrix_cp_vectors.shape[1] + 1,
    )
    return res


def paraDP(N, K, a, b, plot=False, verbose=False, pruning=False):
    """
    Corresponds to [Algorithm 3, Article]
    """
    # T_opt :
    # List of (N + 1) elements
    # Element 0 is unused
    # Each element : matrix of cp vectors with k cps
    T_opt = [np.array([[-1, n - 1]], dtype=np.int) for n in range(0, N + 1)]

    # Z_opt :
    # List of (N + 1) elements
    # Element 0 is unused
    # Each element: list of z breakpoints
    Z_opt = [[-float("inf")]] * (N + 1)
    for k in range(1, K + 1):
        new_T_opt = [T_opt[0]] * (N + 1)
        new_Z_opt = [None] * (N + 1)

        # Case n > k
        for n in range(k + 1, N + 1):

            # Checks
            for m in range(k, n):
                assert utl.isCPvector(T_opt[m], m, k - 1)

            if verbose:
                print(f"n={n}, k={k}")

            if not (pruning and n > k + 1):
                new_T_hat = np.concatenate([expand(T_opt[m], n) for m in range(k, n)])
                if verbose:
                    print(f"T_hat={new_T_hat}")

            else:
                T_bar = full_pruning(
                    n,
                    k,
                    a,
                    b,
                    T_hat.astype(np.int),
                    T_opt[n - 1].astype(np.int),
                    Z_opt[n - 1],
                )
                if verbose:
                    print(f"T_bar={T_bar}")
                new_T_hat = T_bar
            T_hat = np.unique(new_T_hat, axis=0)
            lst_z, lst_t = paraCP(
                n, k, a, b, T_hat.astype(np.int), plot=(k == K and n == N) and plot
            )
            new_T_opt[n] = np.array(lst_t)
            new_Z_opt[n] = lst_z

        T_opt = new_T_opt
        Z_opt = new_Z_opt
    return lst_z, T_opt[N]
