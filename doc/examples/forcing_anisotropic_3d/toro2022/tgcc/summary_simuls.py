"""



"""

from pathlib import Path
from runpy import run_path

import numpy as np


here = Path(__file__).absolute().parent

namespace = run_path(here.parent / "util.py")
couples320 = namespace["couples320"]
couples_focus = [
    (0.66, 18000),
    (3, 900),
    (10, 160),
    (40, 1),
    (40, 20),
]

couples_too_hard = [
    (120, 10),
    (40, 80),
]

couples_need_dns = [
    (40, 20),
]

nb_cores_per_node = 24
memory_per_node = 192  # GB

# here.parent / "occigen / simuls_cannot_be_relaunched.txt"
# here.parent / "occigen / simuls_not_finished.txt"

possible_nz = np.array([480, 528, 672])


possible_nz_versus_nhovernz = {
    2: [528, 672],
    4: [336, 432, 480, 576],
    8: [240, 288, 336],
    16: [96, 192, 288],
}

nb_nodes_maxs = np.arange(1, 12)
memory_nodes = nb_nodes_maxs * memory_per_node
print(f"{memory_nodes=}")

for nhovernz, nzs in possible_nz_versus_nhovernz.items():
    print(79 * "-" + f"\n{nhovernz=}")
    for nz in nzs:
        good = nz / nb_cores_per_node / nb_nodes_maxs % 1 == 0

        nb_nodes_ok = good.nonzero()[0]

        nh = nz * nhovernz
        print(f"{nh=}, {nz=}, {nb_nodes_ok=}")

        # really bad estimation!
        estimation_memory = nh ** 2 * nz * 8 / 1024**3 * (4*2 + 60)
        nb_cores_min = np.ceil(estimation_memory / memory_per_node)

        print(f"{estimation_memory=:.0f} GB, {nb_cores_min=:.0f}")
