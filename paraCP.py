import numpy as np
import matplotlib.pyplot as plt
import utilities as utl


def paraCP(n, k, a, b, T_hat, plot=False):
    """
    Corresponds to [Algorithm 1, Article] and [§2.3.1, Report]
    Given T_hat a list of CP vectors, compute the optimal quadratics T_opt
    """

    u = 0

    # Initialization: at z = - infinity as described in [A.2, Article]
    lst_z = [-float("inf")]
    qfs = [(tau, utl.get_QF(n, k, a, b, tau)) for tau in T_hat]
    poly_coeffs = np.array([qf[1] for qf in qfs])
    lst_t_ind, min_qf = utl.lex_min(utl.invert_deg_one(poly_coeffs))
    lst_t = [qfs[lst_t_ind][0]]

    done = False

    if plot:
        Z_min = -2
        Z_max = 2
        Z = np.linspace(Z_min, Z_max, num=1000)
        plt.figure(figsize=(15, 15))
        plt.ylim(-10, 10)
        for i, qf in enumerate(poly_coeffs):
            plt.plot(Z, utl.eval_poly(qf, Z), label=f"{qf}", color="gray")

    assert not np.isnan(poly_coeffs).any()
    while not done:
        # Apply [Lemma 1, Report]
        differences = poly_coeffs - poly_coeffs[None, lst_t_ind]
        assert (differences[lst_t_ind] == 0).all()
        assert not np.isnan(differences).any()
        ret = utl.next_break_point(*(differences.T), lst_z[-1])
        if ret is None:
            done = True
        else:
            new_lst_t_ind, next_z = ret
            assert new_lst_t_ind != lst_t_ind

            assert np.isclose(
                utl.eval_poly(poly_coeffs[new_lst_t_ind], next_z),
                utl.eval_poly(poly_coeffs[lst_t_ind], next_z),
            )

            lst_t_ind = new_lst_t_ind
            lst_z.append(next_z)
            lst_t.append(qfs[lst_t_ind][0])

    if plot:
        max_of_min = -float("inf")
        min_of_min = float("inf")
        for i in range(len(lst_z)):
            qf = utl.get_QF(n, k, a, b, lst_t[i])
            lb = lst_z[i] if i > 0 else Z_min
            ub = lst_z[i + 1] if i + 1 < len(lst_z) else Z_max
            current_Z = np.linspace(lb, ub, 100)
            evaluated_poly = utl.eval_poly(qf, current_Z)
            max_of_min = max(np.max(evaluated_poly), max_of_min)
            min_of_min = min(np.min(evaluated_poly), min_of_min)
            plt.plot(current_Z, evaluated_poly, label=f"{qf}", color="red")
        plt.ylim(min_of_min - 0.2, max_of_min + 0.7)
        plt.legend()
        plt.show()
    return lst_z, lst_t