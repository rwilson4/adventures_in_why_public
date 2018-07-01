from __future__ import print_function, division
import numpy as np
import math
import sys
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def execute(sf=100, alpha=0.05, Bp=400):
    alpha = 0.05  # p-value threshold
    Bss = 100000  # Number of simulations for statistical significance

    N = 1000 * sf
    Q_Alice = 2 * sf  # Opens email regardless
    Q_Brian = 1 * sf  # Only opens email if receives A
    Q_Charlotte = 3 * sf  # Only opens email if receives B
    Q_George = N - Q_Charlotte - Q_Brian - Q_Alice

    Q = np.array([Q_Alice, Q_Brian, Q_Charlotte, Q_George])
    p = 0.5
    na = int(N / 2)
    nb = N - na

    beta = 0
    k = 1
    for i in range(Bp):
        if i > 0.01 * k * Bp:
            logging.info('{0:d}: {1:.01%}'.format(1000*sf, 0.01 * k))
            k += 1

        qA = multivariate_hypergeometric(Q, na)

        qB = Q - qA
        ea = qA[0] + qA[1]
        eb = qB[0] + qB[2]
        # print(qA.astype(int))
        # print(qB.astype(int))
        # print(int(ea), int(eb))
        # print(int(na), int(nb))

        stat_sig = stat_sig_hypergeometric(sum(qA), sum(qB), ea, eb, alpha, B=Bss)

        if stat_sig:
            # print('Stat sig\n')
            beta += 1
        else:
            pass
            # print('Not stat sig\n')

    power = beta / Bp
    plow, phigh = wilson_ci(beta, Bp - beta)
    fmt = "{0:>11d}\t{1:>11d}\t{2:>10d}\t{3:6.1%}\t{4:9.1%}\t{5:10.1%}"
    print(fmt.format(N, Bp, beta, power, plow, phigh))


def wilson_ci(s, f):
    z = 1.96
    z2 = z * z
    n = s + f

    ctr = (s + z2 / 2) / (n + z2)
    wdth = (z / (n + z2)) * math.sqrt(s * f / n + z2 / 4)
    plow = ctr - wdth
    phigh = ctr + wdth

    return plow, phigh


def multivariate_hypergeometric(Q, n):
    # Strategy from https://stackoverflow.com/questions/35734026/numpy-drawing-from-urn
    q = np.zeros((len(Q),))
    for i in range(len(Q)-1):
        if sum(q) == n:
            return q

        q[i] = np.random.hypergeometric(Q[i], sum(Q[(i+1):]), n - sum(q))

    q[-1] = n - sum(q)
    return q


def stat_sig_hypergeometric(na, nb, ea, eb, alpha, B=1000000):
    N = na + nb
    good = ea + eb
    bad = N - good
    oe_obs = np.abs(eb / nb - ea / na)

    sa = np.random.hypergeometric(good, bad, na, size=B)
    sb = good - sa
    z = np.abs(sb / nb - sa / na)
    pval = np.mean(z >= oe_obs)
    if pval <= alpha:
        return True
    else:
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Power Calculator')
    parser.add_argument('--alpha', type=float,
                        default=0.05,
                        help='p-value threshold')
    parser.add_argument('--simulations', type=int,
                        default=400,
                        help='Number of power simulation to run')

    args = parser.parse_args()
    alpha = args.alpha
    Bp = args.simulations

    sfs = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]

    tmpl = 'Running with alpha={0:.02f}, {1:d} simulations'
    logging.info(tmpl.format(alpha, Bp))

    print("Sample Size\tSimulations\tRejections\tPower\tPower Low\tPower High")
    for sf in sfs:
        execute(sf=sf, alpha=alpha, Bp=Bp)
