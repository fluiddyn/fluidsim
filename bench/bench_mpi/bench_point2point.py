from time import perf_counter
import sys

from mpi4py import MPI
import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print(MPI.Get_library_version())

# 8 bytes
dtype = np.float64

size = 100

times = []
sizes = []

for _ in range(16):
    size *= 2
    sizes.append(size)
    if rank == 0:
        data = np.arange(size, dtype=dtype)
        comm.Send([data, MPI.DOUBLE], dest=1, tag=77)
    elif rank == 1:
        data = np.empty(size, dtype=dtype)
        comm.Recv([data, MPI.DOUBLE], source=0, tag=77)
        assert np.allclose(data, np.arange(size, dtype=dtype))

    comm.barrier()
    t0 = perf_counter()
    if rank == 0:
        comm.Send([data, MPI.DOUBLE], dest=1, tag=77)
    elif rank == 1:
        comm.Recv([data, MPI.DOUBLE], source=0, tag=77)

    comm.barrier()

    duration = perf_counter() - t0

    times.append(duration)

    if rank == 0:
        print(
            f"{duration:.3e} s for {size:8d} floats ({8e-9 * size / duration:.3f} Go/s)"
        )

if rank > 0:
    sys.exit()

print(sizes)
print(times)

poly = Polynomial.fit(sizes, times, 1, window=(min(sizes), max(sizes)))
print(poly)
bandwidth = 8e-9 / poly.coef[1]
print(f"{bandwidth = :.3g} Go/s")

# fig, ax = plt.subplots()
# ax.plot(sizes, times, "ok")
# ax.plot(sizes, poly(np.array(sizes)), "r")
# ax.set_xlabel("number of floats")
# ax.set_ylabel("t (s)")
# ax.set_xscale("log")
# ax.set_yscale("log")
# plt.show()
