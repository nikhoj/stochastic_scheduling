# first we need to find how many configuration can be made
import numpy as np

m = 2
n = 10
J = set(np.arange(0,10,1))
print(J)
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


results = list(powerset(J))
results = results[1:int(len(results)/2)]
print(results)
