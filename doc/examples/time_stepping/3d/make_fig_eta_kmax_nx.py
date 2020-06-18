import numpy as np
import matplotlib.pyplot as plt

from which_params import nx_bigs, nb_procs, deltak, eta, compute

coef_dealiasing = 2 / 3
nx = np.array(nx_bigs)
kmax = coef_dealiasing * deltak * nx / 2

print(f"{kmax * eta = }")

fig, ax = plt.subplots()

ax.plot(nx, kmax * eta, "-x", label="coef truncation = 2/3")

nx_small = []
nx_medium = []
for nx, nb_proc in zip(nx_bigs, nb_procs):
    nxs = compute(nx, nb_proc, verbose=False)
    nx_small.append(nxs[-1])
    nx_medium.append(nxs[1])

kmax = 0.9 * deltak * np.array(nx_medium) / 2
ax.plot(nx_medium, kmax * eta, "-o", label="coef truncation = 0.9")

kmax = deltak * np.array(nx_small) / 2
ax.plot(nx_small, kmax * eta, "-o", label="coef truncation = 1")

ax.set_xlabel("$n_x$")
ax.set_ylabel(r"$k_{\max} \eta$")

ax.legend()
plt.show()
