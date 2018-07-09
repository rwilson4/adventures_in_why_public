from __future__ import print_function, division
import numpy as np
import math
import sys
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def execute(sf=100, alpha=0.05, Bp=400, sample_method='fixed', test_null=False):
    Bss = 100000  # Number of simulations for statistical significance

    N = 1000 * sf
    if test_null:
        logging.info('Testing Type I error rate')
        Q_Alice = 6 * sf  # Opens email regardless
        Q_Brian = 0 * sf  # Only opens email if receives A
        Q_Charlotte = 0 * sf  # Only opens email if receives B
    else:
        logging.info('Testing Type II error rate')
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

        if sample_method == 'fixed':
            qA = multivariate_hypergeometric(Q, na)
        else:
            qA = multivariate_binomial(Q, p)
            na = sum(qA)
            nb = N - na

        qB = Q - qA
        ea = qA[0] + qA[1]
        eb = qB[0] + qB[2]
        # print(qA.astype(int))
        # print(qB.astype(int))
        # print(int(ea), int(eb))
        # print(int(na), int(nb))

        if sample_method == 'fixed':
            stat_sig = stat_sig_hypergeometric(sum(qA), sum(qB), ea, eb, alpha, B=Bss)
        else:
            stat_sig = stat_sig_binomial(sum(qA), sum(qB), ea, eb, p, alpha, B=Bss)

        if stat_sig:
            # print('Stat sig\n')
            beta += 1
        else:
            pass
            # print('Not stat sig\n')

    power = beta / Bp
    plow, phigh = wilson_ci(beta, Bp - beta)
    fmt = "{0:>11d}\t{1:>11d}\t{2:>10d}\t{3:6.2%}\t{4:9.2%}\t{5:10.2%}"
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


def multivariate_binomial(Q, p):
    q = np.zeros((len(Q),))
    for i in range(len(Q)):
        q[i] = np.random.binomial(Q[i], p)

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
        # Check if observation is on the boundary of the critical region
        if ea >= eb:
            ea += 1
            eb -= 1
        else:
            ea -= 1
            eb += 1

        oe_obs = np.abs(eb / nb - ea / na)
        new_pval = np.mean(z >= oe_obs)
        if new_pval > alpha:
            return False
        else:
            prob = (alpha - new_pval) / (pval - new_pval)
            if np.random.uniform() <= prob:
                return True
            else:
                return False


def stat_sig_binomial(na, nb, ea, eb, p, alpha, B=1000000):
    N = na + nb
    good = ea + eb
    bad = N - good
    if nb == 0 or na == 0:
        oe_obs = 0
    else:
        oe_obs = eb / nb - ea / na

    ga = np.random.binomial(good, p, size=B)
    ba = np.random.binomial(bad, p, size=B)
    gb = good - ga
    bb = bad - ba

    pa = np.divide(ga.astype(float), ga + ba, out=np.zeros(ga.size),
                   where=(ga + ba)!=0)
    pb = np.divide(gb.astype(float), gb + bb, out=np.zeros(ga.size),
                   where=(gb + bb)!=0)

    z = pb - pa
    abs_z = np.abs(z)
    pval = np.mean(abs_z >= np.abs(oe_obs))
    if pval <= alpha:
        # Reject the Null Hypothesis.
        return True
    else:
        # We want to make the outcome *slightly* more extreme, and
        # compute the associated p-value. If that p-value is greater
        # than alpha, we do not reject the Null Hypothesis. Otherwise,
        # we accept the Null Hypothesis with a certain probability
        # depending on alpha, the p-value calculated above, and the
        # new p-value based on the slightly-modified outcome
        # extremity.
        #
        # There are 4 parameters we can tweak, either by adding or
        # subtracting one.
        #
        # If eb / nb >= ea / na, the following will make the outcome
        # more extreme:
        #
        # - Adding k to eb, and subtracting k from ea (under the Null
        #   Hypothesis, ea + eb is a constant which we cannot alter).
        # - Subtracting l from nb and adding l to na (under either the
        #   Null or Alternative Hypotheses, na + nb is a constant
        #   which we cannot alter).
        #
        # The outcome extremity of the modified example is:
        #   oe'(k, l) = (eb + k) / (nb - l)
        #                   - (ea - k) / (na + l)
        #
        # The goal is to find values of k and l such that
        #     oe < oe'(k, l),
        # and there are no other values k' and l' satsifying
        #     oe < oe'(k', l') < oe'(k, l)
        #
        # Under what circumstances is oe'(1, 0) < oe'(0, 1)? Are there
        # circumstances in which there exist k, l st.
        # oe'(k, l) < oe'(1, 0) and oe'(k, l) < oe'(0, 1)?
        #
        # (eb + 1) / nb - (ea - 1) / na < eb / (nb - 1) - ea / (na + 1)
        #
        # minimize (eb + k) / (nb - l) - (ea - k) / (na + l)
        # s.t.     k, l \in [0, 1, ...]
        #          0 <= k <= ea
        #          0 <= l <= nb
        #          k + l >= 1

        if oe_obs >= 0:
            # Find the smallest value in z that is strictly greater than oe_obs
            oe_prime = np.min(z[z > oe_obs])
            new_pval = np.mean(abs_z >= oe_prime)
        else:
            # Find the largest value in z that is strictly less than oe_obs
            oe_prime = np.max(z[z < oe_obs])
            new_pval = np.mean(abs_z >= -oe_prime)

        if new_pval > alpha:
            return False
        else:
            prob = (alpha - new_pval) / (pval - new_pval)
            if np.random.uniform() <= prob:
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
    parser.add_argument('--method', choices=['fixed', 'random'],
                        default='fixed',
                        help='Hypergeometric or binomial sampling')
    parser.add_argument('--test_null', action='store_true',
                        help='Test power under null hypothesis')

    args = parser.parse_args()
    alpha = args.alpha
    Bp = args.simulations
    method = args.method

    if args.test_null:
        sfs = [1]
    else:
        sfs = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]

    tmpl = 'Running with alpha={0:.02f}, {1:d} simulations, method {2:s}'
    logging.info(tmpl.format(alpha, Bp, method))

    print("Sample Size\tSimulations\tRejections\tPower\tPower Low\tPower High")
    for sf in sfs:
        execute(sf=sf, alpha=alpha, Bp=Bp,
                sample_method=method,
                test_null=args.test_null)
