from pathlib import Path
from itertools import product
from copy import deepcopy
from math import pi
from re import A
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import subprocess
from pathlib import Path


path_base = Path("/scratch/vlabarre/aniso/")
path_output_papermill = Path("/scratch/vlabarre/aniso/results_papermill")

path_base_azzurra = Path("/workspace/vlabarre/aniso/")
path_output_papermill_azzurra = Path(
    "/workspace/vlabarre/aniso/results_papermill"
)

path_base_jeanzay = Path("/gpfsscratch/rech/uzc/uey73qw/aniso/")
path_output_papermill_jeanzay = Path(
    "/gpfsscratch/rech/uzc/uey73qw/aniso/results_papermill"
)


def list_paths(N, Rb, nh, nz, proj="all"):
    # Find the paths of the simulations corresponding to proj, N, R, nh, and nz
    paths = sorted(path_base.glob(f"ns3d.strat_polo*_{nh}x{nh}*"))

    Fh = 1.0 / N
    pathstemp = [
        p for p in paths if f"_Rb{Rb:.3g}_" in p.name and f"_Fh{Fh:.3e}" in p.name
    ]

    if proj == "None":
        paths = [p for p in pathstemp if f"_proj_" not in p.name]
    elif proj == "poloidal":
        paths = [p for p in pathstemp if f"_proj_" in p.name]
    elif proj == "all":
        paths = pathstemp
    else:
        print(f"Projection {proj} not known")
        paths = None

    print(
        f"List of paths for simulations with (N, Rb, nh, nz, proj)= ({N}, {Rb}, {nh}, {nz}, {proj}): \n"
    )

    for path in paths:
        print(path, "\n")

    return paths


def list_paths_jeanzay(N, Rb, nh, nz, proj="all"):
    # Find the paths of the simulations corresponding to proj, N, R, nh, and nz
    paths = sorted(path_base_jeanzay.glob(f"ns3d.strat_polo*_{nh}x{nh}*"))

    Fh = 1.0 / N
    pathstemp = [
        p for p in paths if f"_Rb{Rb:.3g}_" in p.name and f"_Fh{Fh:.3e}" in p.name
    ]

    if proj == "None":
        paths = [p for p in pathstemp if f"_proj_" not in p.name]
    elif proj == "poloidal":
        paths = [p for p in pathstemp if f"_proj_" in p.name]
    elif proj == "all":
        paths = pathstemp
    else:
        print(f"Projection {proj} not known")
        paths = []

    print(
        f"List of paths for simulations with (N, Rb, nh, nz, proj)= ({N}, {Rb}, {nh}, {nz}, {proj}): \n"
    )

    for path in paths:
        print(path, "\n")
    return paths


def compute_nhmin_nhmax(N, R):
    """
    Compute:
    The minimal horizontal resolution nhmin to obtain R4 > 100 (to have a non viscous regime for initial simulation)
    The maximal horizontal resolution nhmax to obtain a DNS: kmax eta = 1.0
    """
    coef_nu4 = 1.0
    Lh = 3.0
    Pf = 1.0
    coef_dealiasing = 0.8
    # Note: kmax = coef_dealiasing * (nh 2 pi) / (2 Lh)
    # Note: eta = pf**(1/2) / (R**3 * N**6)**(1/4)

    nhmin = Lh * (100 * coef_nu4) ** (3 / 10) * N ** (6 / 5) * Pf ** (1 / 10) / pi
    nhmax = (
        Lh
        * (R**3 * N**6) ** (1 / 4)
        / (coef_dealiasing * pi * (Pf ** (1 / 2)))
    )

    return nhmin, nhmax


def plot_nhmin_nhmax_vs_N(Nmin, Nmax, R, save_fig=False):
    N = np.linspace(Nmin, Nmax, 200)
    nhmin, nhmax = compute_nhmin_nhmax(N, R)

    fig, ax = plt.subplots()
    plt.plot(N, nhmin, "b-", label=r"$n_{h,min} ~ (R_4 = 100)$")
    plt.plot(N, nhmax, "r-", label=r"$n_{h,max} ~ (k_{max} \eta = 1)$")
    ax.set_xlim([Nmin, Nmax])
    ax.set_ylim([0, 5120])
    ax.set_title(f"$R = {R}$")
    ax.set_yticks([320 * n for n in range(17)])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$n_h$")
    ax.grid(True)
    ax.minorticks_on()
    ax.legend(loc="upper left")

    if save_fig == True:
        plt.savefig(f"nhmin_nhmax_vs_N_R{R}.eps")

    return ax


def plot_nhmin_nhmax_vs_R(Rmin, Rmax, N, save_fig=False):
    R = np.linspace(Rmin, Rmax, 200)
    nhmin = np.zeros(len(R))
    nhmax = np.zeros(len(R))
    for n in range(len(R)):
        nhmin[n], nhmax[n] = compute_nhmin_nhmax(N, R[n])

    fig, ax = plt.subplots()
    plt.plot(R, nhmin, "b-", label=r"$n_{h,min} ~ (R_4 = 100$)")
    plt.plot(R, nhmax, "r-", label=r"$n_{h,max} ~ (k_{max} \eta = 1)$")
    ax.set_xlim([Rmin, Rmax])
    ax.set_ylim([0, 5120])
    ax.set_title(f"$N = {N}$")
    ax.set_yticks([320 * n for n in range(17)])
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$n_h$")
    ax.grid(True)
    ax.minorticks_on()
    ax.legend(loc="upper left")

    if save_fig == True:
        plt.savefig(f"nhmin_nhmax_vs_R_N{N}.eps")

    return ax


def get_ratio_nh_nz(N):
    "Get the ratio nh/nz"
    if N in [80, 100, 120]:
        return 8
    elif N in [20, 30, 40, 60]:
        return 4
    elif N <= 15:
        return 2
    else:
        raise NotImplementedError


def type_fft_from_nh_nz(nh, nz):
    "Get the fft type to use for a given nh and N"
    if nh == 640:
        return "fftw1d"
    if nh == 1280:
        return "fftw1d"
    if nh == 1920:
        if nz in [120, 240, 480]:
            return "p3dfft"
        if nz == 960:
            return "fftw1d"
    if nh == 2560:
        return "p3dfft"


def nb_nodes_from_nh_nz(nh, nz):
    if nh == 640:
        return 1
    if nh == 1280:
        return nz // 40
    if nh == 1920:
        if nz in [120, 240]:
            return 12
        if nz in [480, 960]:
            return 24
    if nh == 2560:
        return min(nz // 10, 32)


def get_t_end(N, nh):
    "Get end time of the simulation with buoyancy frequency N and horizontal resolution nh"
    if nh == 320:
        return max(20.0, 20.0 * N / 10.0)
    elif nh == 640:
        return max(25.0, 25.0 * N / 10.0)
    elif nh == 1280:
        return max(25.0, 25.0 * N / 10.0) + 5
    elif nh == 1920:
        return max(25.0, 25.0 * N / 10.0) + 9
    elif nh == 2560:
        return max(25.0, 25.0 * N / 10.0) + 11.5
    else:
        raise NotImplementedError


def get_t_statio(N, nh):
    "Get end time of the simulation with buoyancy frequency N and horizontal resolution nh"
    t_end = get_t_end(N, nh)
    if nh < 2560:
        return t_end - 3.0
    else:
        return t_end - 2.0 


def compute_nhstart_nhend(N, R):
    """
    Compute:
    The starting horizontal resolution nhstart is fixed to 320 to developp large scales in small simulations
    The ending horizontal resolution nhend to obtain a DNS: kmax eta > 1.0 and such that nz is a multiple of 40 (number of procs per node on clusters)
    """

    nhmin, nhmax = compute_nhmin_nhmax(N, R)
    ratio_nh_nz = get_ratio_nh_nz(N)
    nzmax = nhmax / ratio_nh_nz
    nzend = ((nzmax // 40) + 1) * 40
    nhend = nzend * ratio_nh_nz

    nhstart = 320

    return nhstart, nhend


def lprod(a, b):
    return list(product(a, b))


couplestarget = set(
    lprod([10, 20, 40], [5, 10, 20, 40, 80, 160])
    + lprod([10, 20, 40, 80, 120], [0.5, 1, 2])
    + lprod([30], [10, 20, 40])
    + lprod([60], [10, 20])
    + lprod([6.5], [100, 200])
    + lprod([4], [250, 500])
    + lprod([3], [450, 900])
    + lprod([2], [1000, 2000])
    + lprod([0.66], [9000, 18000])
    + [(14.5, 20), (5.2, 150), (2.9, 475), (1.12, 3200), (0.25, 64000)]
)

couplestarget.add((80, 10))
couplestarget.add((100, 10))
couplestarget.add((120, 10))
couplestarget.remove((40, 160))


def filter_couples(couplestarget, nh):
    """
    Return a copy of couplestarget where couples such that nhstart > nh or nhend < nh are removed
    """

    couplesnh = deepcopy(couplestarget)
    toremove = []
    for NR in couplesnh:
        N = NR[0]
        R = NR[1]
        nhstart, nhend = compute_nhstart_nhend(N, R)
        if nhstart > nh or nhend < nh:
            toremove.append(NR)

    for NR in toremove:
        couplesnh.remove(NR)

    return couplesnh


couples320 = couplestarget
couples640 = filter_couples(couplestarget, 640)
couples1280 = filter_couples(couplestarget, 1280)
couples1280.add((0.66, 18000))
couples1280.add((2, 2000))
couples1280.add((3, 900))
couples1280.add((4, 500))
couples1280.add((6.5, 200))
couples1280.add((10, 80))
couples1280.add((40, 5))

couples1920 = filter_couples(couplestarget, 1920)
couples1920.add((10, 160))

couples2560 = filter_couples(couplestarget, 2560)

couples = {
    320: couples320,
    640: couples640,
    1280: couples1280,
    1920: couples1920,
    2560: couples2560,
}


# TODO: rather put this function in fluiddyn.fluiddyn.clusters.slurm
def is_job_submitted(name_run):
    command = f"squeue -n {name_run}"
    out = subprocess.check_output(command, shell=True)
    length = len(out.splitlines())
    if length > 1:
        # In this case squeue's output contain at least one job
        return True
    else:
        return False
