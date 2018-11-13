from __future__ import print_function, division
import numpy as np
import sys
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def execute(n, s, alpha, variants, null_hypothesis=None):
    na = n[0]
    nb = n[1]
    ea = s[0]
    eb = s[1]
    if null_hypothesis is None:
        pval = stat_sig_hypergeometric(na, nb, ea, eb)
    else:
        pval = stat_sig_multihypergeometric(na, nb, ea, eb, null_hypothesis)
        
    print('p-value:', pval)


def multivariate_hypergeometric(Q, n, B=1):
    # Strategy from https://stackoverflow.com/questions/35734026/numpy-drawing-from-urn
    q = np.zeros((len(Q),B))
    for j in range(B):
        for i in range(len(Q)-1):
            if sum(q[:, j]) == n:
                break

            q[i, j] = np.random.hypergeometric(Q[i], sum(Q[(i+1):]), n - sum(q[:, j]))
        else:
            q[len(Q)-1, j] = n - sum(q[:, j])
    return q


def stat_sig_hypergeometric(na, nb, ea, eb, B=1000000):
    N = na + nb
    good = ea + eb
    bad = N - good
    oe_obs = np.abs(eb / nb - ea / na)

    sa = np.random.hypergeometric(good, bad, na, size=B)
    sb = good - sa
    z = np.abs(sb / nb - sa / na)
    pval = np.mean(z >= oe_obs)
    return pval


def stat_sig_multihypergeometric(na, nb, ea, eb, null_hypothesis, B=1000000):
    expected = (null_hypothesis[1] + null_hypothesis[3]) / (na + nb)
    expected -= (null_hypothesis[1] + null_hypothesis[2]) / (na + nb)
    oe_obs = np.abs(eb / nb - ea / na - expected)
    q = multivariate_hypergeometric(null_hypothesis, na, B=B)
    sa = q[1, :] + q[2, :]
    sb = (null_hypothesis[1] - q[1, :]) + (null_hypothesis[3] - q[3, :])
    z = np.abs(sb / nb - sa / na - expected)
    pval = np.mean(z >= oe_obs)
    return pval


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Confidence Intervals')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='p-value threshold')
    parser.add_argument('--group-sizes', type=int, nargs=2,
                        required=True,
                        help='Group sizes')
    parser.add_argument('--successes', type=int, nargs=2,
                        help='Number of successes in each group')
    parser.add_argument('--variants', nargs=2,
                        help='Names for variants')

    parser.add_argument('--sample', action='store_true',
                        help='Sample from MHG distribution')
    parser.add_argument('--null-hypothesis', type=int, nargs=4,
                        help='Null Hypothesis specification')

    args = parser.parse_args()
    if args.variants is None:
        variants = ['A', 'B']
    else:
        variants = args.variants

    if args.sample:
        if args.null_hypothesis is None:
            if args.successes is None:
                raise SyntaxError('Please specify a null hypothesis')

            Q = [0, 0, 0, 0]
            Q[0] = sum(args.group_sizes) - sum(args.successes)
            Q[1] = sum(args.successes)
        else:
            Q = args.null_hypothesis

        n = args.group_sizes[0]
        q = multivariate_hypergeometric(Q, n).squeeze()
        ea = q[1] + q[2]
        eb = (Q[1] - q[1]) + (Q[3] - q[3])
        print(q)
        print(ea, eb)
        oe = np.abs(ea / n - eb / (sum(Q) - n))
        print('{0:.02%}'.format(oe))
    else:
        execute(args.group_sizes, args.successes, args.alpha, variants,
                null_hypothesis=args.null_hypothesis)
