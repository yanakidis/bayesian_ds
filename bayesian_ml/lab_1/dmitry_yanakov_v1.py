import numpy as np
from scipy.stats import binom, poisson


def pa(params, model):
    n = params['amax'] - params['amin'] + 1

    prob = np.full(n, 1 / n)
    val = np.arange(params['amin'], params['amax'] + 1)

    return prob, val


def pb(params, model):
    n = params['bmax'] - params['bmin'] + 1

    prob = np.full(n, 1 / n)
    val = np.arange(params['bmin'], params['bmax'] + 1)

    return prob, val


def pc_ab(a, b, params, model):
    val = np.arange(params['amax'] + params['bmax'] + 1)  # c

    if model == 1:
        prob = np.zeros((len(val), len(a), len(b)))

        bin_a_p1 = binom.pmf(np.arange(params['amax'] + 1)[:, np.newaxis],
                             a, params['p1']).T
        bin_b_p2 = binom.pmf(np.arange(params['bmax'] + 1)[:, np.newaxis],
                             b, params['p2']).T

        for i_1 in range(len(bin_a_p1)):
            for i_2 in range(len(bin_b_p2)):
                prob[:, i_1, i_2] = np.convolve(bin_a_p1[i_1, :], bin_b_p2[i_2, :])

    else:  # model == 2
        lambdas = a[:, np.newaxis] * params['p1'] + b * params['p2']

        prob = poisson.pmf(val[:, np.newaxis, np.newaxis], lambdas)

    return prob, val


def pc(params, model):
    a_prob, a_val = pa(params, model)
    b_prob, b_val = pb(params, model)
    c_ab_prob, val = pc_ab(a_val, b_val, params, model)

    prob = np.sum(c_ab_prob, axis=(1, 2)) * a_prob[0] * b_prob[0]  # p(a), p(b) - const

    return prob, val


def pd_c(c, params, model):
    val = np.arange(0, 2 * (params['amax'] + params['bmax']) + 1)  # d

    prob = binom.pmf(np.arange(0, 2 * (params['amax'] + params['bmax']) + 1)[:, np.newaxis],
                     c, params['p3'], c)

    return prob, val


def pd(params, model):
    c_prob, c_val = pc(params, model)
    d_c_prob, val = pd_c(c_val, params, model)

    prob = np.sum(d_c_prob * c_prob, axis=1)

    return prob, val


def pc_a(a, params, model):
    c_ab_prob, _ = pc_ab(a, np.arange(params['bmin'], params['bmax'] + 1), params, model)
    b_prob, _ = pb(params, model)

    prob = np.sum(c_ab_prob, axis=2) * b_prob[0]
    val = np.arange(params['amax'] + params['bmax'] + 1)

    return prob, val


def pc_b(b, params, model):
    c_ab_prob, _ = pc_ab(np.arange(params['amin'], params['amax'] + 1), b, params, model)
    a_prob, _ = pa(params, model)

    prob = np.sum(c_ab_prob, axis=1) * a_prob[0]
    val = np.arange(params['amax'] + params['bmax'] + 1)

    return prob, val


def pc_d(d, params, model):
    d_c_prob = pd_c(np.arange(params['amax'] + params['bmax'] + 1), params, model)[0][d]
    c_prob, val = pc(params, model)
    d_prob = pd(params, model)[0][d]

    prob = (d_c_prob * c_prob.T).T / d_prob

    return prob, val


def pc_abd(a, b, d, params, model):
    d_c_prob, _ = pd_c(np.arange(params['amax'] + params['bmax'] + 1), params, model)
    d_c_prob = d_c_prob[d].T[:, :, np.newaxis, np.newaxis]

    c_ab_prob, val = pc_ab(a, b, params, model)

    a_prob, _ = pa(params, model)
    b_prob, _ = pb(params, model)

    abcd_prob = d_c_prob * c_ab_prob[:, np.newaxis, :] * a_prob[0] * b_prob[0]
    abcd_prob = np.swapaxes(abcd_prob, 1, 2)
    abcd_prob = np.swapaxes(abcd_prob, 2, 3)

    prob = abcd_prob / np.sum(abcd_prob, axis=0)

    return prob, val
