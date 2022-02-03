from time import perf_counter
import pstats
import cProfile

import pyfftw
import fluidfft
import matplotlib.pyplot as plt

t0 = perf_counter()

cProfile.runctx(
    "import fluidsim.solvers.ns3d.output.spatiotemporal_spectra",
    globals(),
    locals(),
    "profile.pstats",
)
t_end = perf_counter()

path = "profile.pstats"
s = pstats.Stats(path)
# s.strip_dirs().sort_stats('time').print_stats(16)
s.sort_stats("time").print_stats(12)
print(
    "with gprof2dot and graphviz (command dot):\n"
    f"gprof2dot -f pstats {path} | dot -Tpng -o profile.png"
)
