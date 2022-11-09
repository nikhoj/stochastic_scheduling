import numpy as np
import itertools

# --------- input -------------
m = 5

# ------------- Vector generate here ---------------------
vec_A = np.arange(1, m + 1, 1)
vec_B = np.arange(-1, -m - 1, -1)
vec_B = np.array(list(itertools.permutations(list(vec_B))))

vec_A = np.tile(vec_A, (len(vec_B), 1))

sum_A_B = vec_A + vec_B

table = []
for i in range(len(sum_A_B)):
    if np.all(np.less_equal(np.absolute(sum_A_B[i]), [m / 2] * m)):
        table.append(sum_A_B[i])
table = np.array(table)

print(table)
