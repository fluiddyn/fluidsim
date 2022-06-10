"""



"""

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

from fluidsim.util import get_mean_values_from_path, times_start_last_from_path


here = Path(__file__).absolute().parent

util_main = run_path(here.parent / "util.py")
couples320 = util_main["couples320"]
path_base = util_main["path_base"]
customize = util_main["customize"]

couples_focus = [
    (0.66, 18000),
    (3, 900),
    (10, 160),
    (40, 1),
    (40, 20),
]

couples_too_hard = [(120, 10), (80, 10), (40, 80), (20, 160)]

couples_need_dns = [
    (40, 20),
]

targets_vs_couple = {(40, 80): 0.3, (40, 40): 0.5, (80, 10): 0.5}


nb_cores_per_node = 24
memory_per_node = 192  # GB

with open(here.parent / "occigen/simuls_cannot_be_relaunched.txt", "r") as file:
    names_cannot_be_relaunched = set([line.strip() for line in file])

with open(here.parent / "occigen/simuls_not_finished.txt", "r") as file:
    names_not_finished = set([line.strip() for line in file])

names_cannot_be_relaunched.update(names_not_finished)


possible_nz_versus_nhovernz = {
    2: [480, 528, 672, 768],
    4: [336, 432, 480, 576],
    8: [240, 288, 336],
    16: [96, 192, 288],
}

nb_nodes_maxs = np.arange(1, 25)
memory_nodes = nb_nodes_maxs * memory_per_node
print(f"{memory_nodes=}")

for nhovernz, nzs in possible_nz_versus_nhovernz.items():
    print(79 * "-" + f"\n{nhovernz=}")
    for nz in nzs:
        nb_points_per_process = nz / (nb_cores_per_node * nb_nodes_maxs)
        good = nb_points_per_process % 1 == 0
        nb_nodes_ok = good.nonzero()[0] + 1

        nh = nz * nhovernz
        print(f"{nh=}, {nz=}, {nb_nodes_ok=}")

        # really bad estimation!
        estimation_memory = nh**2 * nz * 8 / 1024**3 * (4 * 2 + 60)
        nb_cores_min = np.ceil(estimation_memory / memory_per_node)

        print(f"{estimation_memory=:.0f} GB, {nb_cores_min=:.0f}")


path_base_occigen = path_base / "from_occigen/aniso"
paths_all = sorted(
    list(path_base.glob("aniso/ns3d.strat*"))
    + list(path_base_occigen.glob("ns3d.strat*"))
)


new_simuls = []


for couple in sorted(couples320):
    N, Rb = couple
    str_N = f"_N{N}_"
    str_Rb = f"_Rb{Rb:.3g}_"
    paths_couple = [
        p
        for p in paths_all
        if str_N in p.name
        and str_Rb in p.name
        and p.name not in names_cannot_be_relaunched
    ]
    paths_couple.sort(key=lambda p: int(p.name.split("x")[1]))
    path_better = paths_couple[-1]

    name = path_better.name

    values = get_mean_values_from_path(
        path_better, tmin="t_start+2", customize=customize
    )
    assert name == values["name"]

    particular_case = ""
    target_kmax_eta = 0.65

    if couple in couples_need_dns:
        target_kmax_eta = 1.0
        particular_case = "needs DNS"
    elif couple in couples_focus:
        target_kmax_eta = 0.95
        particular_case = "focus"
    elif couple in couples_too_hard:
        target_kmax_eta = 0.34
        particular_case = "too hard"

    if couple in targets_vs_couple:
        target_kmax_eta = targets_vs_couple[couple]
        particular_case = "special"

    kmax_eta = values["k_max*eta"]
    if kmax_eta > target_kmax_eta:
        continue

    nx = values["nx"]
    nz = values["nz"]
    nhovernz = nx // nz
    nz_target = np.ceil(nz * target_kmax_eta / kmax_eta)

    nz_new_simuls = np.array(possible_nz_versus_nhovernz[nhovernz])

    larger_than_target = nz_new_simuls > nz_target
    if any(larger_than_target):
        nz_max = nz_new_simuls[np.argmax(nz_new_simuls > nz_target)]
    else:
        nz_max = nz_new_simuls[-1]
    nz_new_simuls = nz_new_simuls[nz_new_simuls <= nz_max]
    nz_new_simuls = nz_new_simuls[nz_new_simuls > 1.2 * nz]

    # # super special case for couple == (30, 40)
    if 432 in nz_new_simuls and 480 in nz_new_simuls:
        nz_new_simuls = [n for n in nz_new_simuls if n != 432]

    nhnz_new_simuls = [(n * nhovernz, n) for n in nz_new_simuls]

    print(f"--------\n{N=}, {Rb=}\n{path_better.name}")
    print(
        f"{nx=}, {nz=}, {kmax_eta=:.2f}, {target_kmax_eta=:.2f} {particular_case}"
    )
    print(f"{nz_target=}\n{nhovernz=}\n{nhnz_new_simuls}")

    t_start_better, t_last_better = times_start_last_from_path(path_better)

    t_start = round(t_last_better, 1)

    nz_better = nz
    init = name[28:]
    for nh, nz in nhnz_new_simuls:

        if nh > 2000:
            t_simul = 2.5
        else:
            t_simul = 4.0

        new_simuls.append(
            dict(
                N=N,
                R_i=Rb,
                nh=nh,
                nz=nz,
                nhovernz=nhovernz,
                t_start=t_start,
                t_end=t_start + t_simul,
                kmax_eta=kmax_eta * nz / nz_better,
                init=init,
            )
        )
        t_start += t_simul
        init = (nh, nz)

new_simuls = pd.DataFrame(new_simuls)

print(new_simuls)
