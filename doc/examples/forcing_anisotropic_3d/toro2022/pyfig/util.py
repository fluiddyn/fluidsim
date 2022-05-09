from pathlib import Path
import os
from itertools import product
import sys

import matplotlib.pyplot as plt

from fluidsim.util import times_start_last_from_path

path_base = os.environ["STRAT_TURB_TORO2022"]

paths_all = sorted(Path(path_base).glob("simul_folders/ns3d*"))

here = Path(__file__).absolute().parent
tmp_dir = here / "tmp"
tmp_dir.mkdir(exist_ok=True)


height = 3.7
plt.rc("figure", figsize=(1.33 * height, height))


def get_path_finer_resol(N, Rb):
    str_N = f"_N{N}_"
    str_Rb = f"_Rb{Rb:.3g}_"
    str_Rb2 = f"_Rb{Rb}_"
    paths_couple = [
        p
        for p in paths_all
        if str_N in p.name and (str_Rb in p.name or str_Rb2 in p.name)
    ]
    paths_couple.sort(key=lambda p: int(p.name.split("x")[1]), reverse=True)
    for path in paths_couple:
        t_start, t_last = times_start_last_from_path(path)
        if t_last > t_start + 1:
            return path


def lprod(a, b):
    return list(product(a, b))


couples320 = set(
    lprod([10, 20, 40], [5, 10, 20, 40, 80, 160])
    + lprod([30], [10, 20, 40])
    + lprod([6.5], [100, 200])
    + lprod([4], [250, 500])
    + lprod([3], [450, 900])
    + lprod([2], [1000, 2000])
    + lprod([0.66], [9000, 18000])
    + [(14.5, 20), (5.2, 150), (2.9, 475), (1.12, 3200), (0.25, 64000)]
)

couples320.add((80, 10))
couples320.add((120, 10))
couples320.remove((40, 160))

# Small Rb
couples320.update(lprod([20], [1, 2]))
couples320.update(lprod([40], [1, 2]))
couples320.update(lprod([80], [0.5, 1]))

has_to_save = "SAVE" in sys.argv


def save_fig(fig, name):
    if has_to_save:
        fig.savefig(tmp_dir / name)
