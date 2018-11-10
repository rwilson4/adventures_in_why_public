from __future__ import division, print_function
import numpy as np

N  = 1000      # Audience size
n  =  500      # Experiment group size
ea =  100      # Email opens in group A
eb =  105      # Email opens in group B
good = ea + eb # Total email opens
bad = N - good # Number of non-email openers

# Observed outcome extremity
oe_obs = np.abs(eb / (N - n) - ea / n)

B = 1000000 # Number of simulations
p = 0
for i in range(B):
    # Simulate how many email opens we might randomly sample
    sa = np.random.hypergeometric(good, bad, n)

    # Number of email openers who are necessarily in the second group
    sb = good - sa

    # Simulated outcome extremity
    oe_sim = np.abs(sb / (N - n) - sa / n)

    if oe_sim >= oe_obs:
       p += 1

pval = p / B
print(pval)
