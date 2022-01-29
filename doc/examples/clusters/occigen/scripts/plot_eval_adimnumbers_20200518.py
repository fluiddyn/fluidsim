import numpy as np
import matplotlib.pyplot as plt

nu = 1e-6
Lf = 6.5

Ns = [0.1]  # , 0.5, 1.0, 4]
diameters = [0.25, 0.5]
velocities = [0.1, 0.2, 0.5]

fig, ax = plt.subplots()

counter_simul = 0

for N in Ns:
    for diameter in diameters:
        for speed in velocities:
            mesh = 3 * diameter
            epsilon = 0.35 * speed**3 / mesh * diameter / Lf
            R = epsilon / (nu * N)
            U2 = 0.1 * speed**2
            Fh = epsilon / (U2 * N)

            # if R < 1 or Fh > 0.2:
            #     continue

            counter_simul += 1

            marker = "x"
            if diameter == 0.5:
                marker = "o"

            sc = ax.scatter(Fh, R, c=N, vmin=min(Ns), vmax=max(Ns), marker=marker)

print(f"{counter_simul} simulations")

plt.colorbar(sc)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$F_h$")
ax.set_ylabel(r"$\mathcal{R}$")
plt.show()
