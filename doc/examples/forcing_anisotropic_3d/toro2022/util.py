from pathlib import Path
from itertools import product
from copy import deepcopy
from math import sqrt

from fluiddyn.clusters.legi import Calcul8 as C

cluster = C()

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

cluster.commands_setting_env = [
    "source /etc/profile",
    ". $HOME/miniconda3/etc/profile.d/conda.sh",
    "conda activate env_fluidsim",
    f"export FLUIDSIM_PATH={path_base}",
]


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

couples640 = deepcopy(couples320)
couples640.remove((10, 5))


def get_ratio_nh_nz(N):
    "Get the ratio nh/nz"
    if N >= 80:
        return 16
    elif N == 40:
        return 8
    elif N in [20, 30]:
        return 4
    elif N <= 15:
        return 2
    else:
        raise NotImplementedError


couples896 = set(
    lprod([10], [80, 160])
    + lprod([20], [20, 40, 80])
    + lprod([40], [10, 20, 40, 80])
    + lprod([80, 120], [10])
)

couples1344 = set(
    lprod([10], [160])
    + lprod([20], [40, 80])
    + lprod([40], [10, 20, 40, 80])
    + lprod([80, 120], [10])
)

couples1792 = set(
    lprod([20], [40, 80]) + lprod([40], [10, 20, 40]) + lprod([80, 120], [10])
)
couples2240 = set(
    lprod([20], [80]) + lprod([40], [20, 40]) + lprod([80, 120], [10])
)

couples = {
    320: couples320,
    640: couples640,
    896: couples896,
    1344: couples1344,
    1792: couples1792,
    2240: couples2240,
}


def customize(result, sim):

    EKh = result["EKh"]
    EKz = result["EKz"]
    EK = EKh + EKz
    U = sqrt(2 * EK / 3)
    nu_2 = sim.params.nu_2
    epsK = result["epsK"]

    result["name"] = sim.output.name_run

    result["lambda"] = sqrt(U**2 * nu_2 / epsK)
    result["Re_lambda"] = U * result["lambda"] / nu_2

    result["Rb"] = float(sim.params.short_name_type_run.split("_Rb")[-1])
    result["nx"] = sim.params.oper.nx
    result["nz"] = sim.params.oper.nz
