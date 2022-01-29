#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

import matplotlib.pyplot as plt


f = 1.0
c = 1.0

L_D = c / f

Lx = 30.0
nx = 64
nkx = nx / 2.0 + 1

kx = 2 * np.pi / Lx * np.arange(nkx)

omega_theo = np.sqrt(f**2 + (c * kx) ** 2)

# numerical results
k_num = np.array([0.63, 1.88, 2.51, 4.19])

frequency = np.array([0.1875, 0.34, 0.43, 0.7])


ikx = 14


fig = plt.figure(2)
fig.clf()
plt.hold(True)

plt.plot(kx * L_D, omega_theo / (2 * np.pi), "k", linewidth=2)

plt.plot(kx * L_D, kx * L_D / (2 * np.pi), "k--", linewidth=0.5)

plt.plot(kx[ikx] * L_D, omega_theo[ikx] / (2 * np.pi), "rx", linewidth=2)

plt.plot(k_num * L_D, frequency, "bo", linewidth=2)


plt.xlabel("k L_D")
plt.ylabel("frequency")
plt.show()
