"""
Needs some memory:

```
srun --pty --ntasks=1 --cpus-per-task=4 --hint=nomultithread --time=0:15:0 bash
conda activate env_fluidsim
python postrun1280.py
```

For each finished simulation:

1. clean up the directory
2. prepare a file with larger resolution
3. compute the spatiotemporal spectra
4. execute and save a notebook analyzing the simulation

"""

from pathlib import Path
from shutil import copyfile, rmtree
import re
import subprocess
import os
from itertools import product

import papermill as pm

from fluiddyn.util import modification_date
from fluidsim.util import times_start_last_from_path, load_params_simul
from fluidsim import load

from util import (
    path_base_jeanzay,
    path_output_papermill_jeanzay,
    get_t_end,
    get_t_statio,
    couples1280,
)

from fluidjean_zay import cluster

nh = 1280
deltat = 0.1

path_output_papermill_jeanzay.mkdir(exist_ok=True)

path_end_states = path_base_jeanzay / "end_states"
path_end_states.mkdir(exist_ok=True)

path_init_jeanzay = path_base_jeanzay / "init_jeanzay"
path_init_jeanzay.mkdir(exist_ok=True)

paths = sorted(path_base_jeanzay.glob("ns3d*_polo_*_1280x1280x*"))
paths1920 = sorted(path_base_jeanzay.glob("ns3d*_polo_*_1920x1920x*"))

for path in paths:
    t_start, t_last = times_start_last_from_path(path)

    params = load_params_simul(path)
    N = float(params.N)
    Rb = float(re.search(r"_Rb(.*?)_", path.name).group(1))
    nx = params.oper.nx
    proj = params.projection
    t_end = get_t_end(N, nh)
    t_statio = get_t_statio(N, nh)

    # Simulations with nu = 0 where just for testing on Licallo
    if params.nu_2 == 0.0 or N == 120:
        print(f"{path.name:90s} corresponds to a simulation with nul viscosity)")
        continue

    if t_last < t_end:
        print(f"{path.name:90s} not finished ({t_last=})")
        continue
    print(f"{path.name:90s} done ({t_last=})")

    # delete some useless restart files
    params = load_params_simul(path)
    deltat_file = params.output.periods_save.phys_fields
    path_files = sorted(path.glob(f"state_phys*"))
    for path_file in path_files:
        time = float(path_file.name.rsplit("_t", 1)[1][:-3])
        if (
            # time % deltat_file > deltat
            time != t_last
            and abs(time - t_end) > deltat
        ):
            print(f"deleting {path_file.name}")
            path_file.unlink()

        if abs(time - t_end) < deltat:
            path_end_state = path_file
            link_last_state = path_end_states / path.name / path_end_state.name
            if not link_last_state.exists():
                link_last_state.parent.mkdir(exist_ok=True)
                link_last_state.symlink_to(path_end_state)

    # compute spatiotemporal spectra
    sim = load(path, hide_stdout=True)
    sim.output.spatiotemporal_spectra.get_spectra(tmin=t_statio)

    Fh = 1.0 / N
    Fh_str = "_Fh" + "{Fh:.3e}"
    Rb_str = "_Rb" + "{Rb:.3g}"

    try:
        corresponding_path1920 = next(
            p for p in paths1920 if Fh_str in p.name and Rb_str in p.name
        )
    except StopIteration:
        corresponding_path1920 = False

    try:
        path_init_file = next(path.glob("State_phys_1920x1920*/state_phys_t*.h5"))
    except StopIteration:
        path_init_file = False

    if not corresponding_path1920:
        if not path_init_file:
            name_run = f"modif_reso_polo_nx{nh}"
            command = (
                f"srun fluidsim-modif-resolution --t_approx {t_end} {path} 1.5"
            )
            print(f"run command: {command}\n")
            cluster.submit_command(
                f"{command}",
                name_run=name_run,
                nb_nodes=1,
                nb_cores_per_node=20,
                nb_cpus_per_task=20,
                nb_tasks_per_node=1,
                nb_tasks=1,
                nb_mpi_processes=1,
                omp_num_threads=1,
                ask=True,
                walltime="00:15:00",
                project="uzc@cpu",
                partition="prepost",
            )

    else:
        if path_init_file:
            path_init_dir = path_init_file.parent
            print(f"deleting {path_init_dir}")
            rmtree(path_init_dir, ignore_errors=True)

    path_in = "analyse_1simul_papermill.ipynb"
    path_ipynb = path_out = (
        path_output_papermill_jeanzay
        / f"analyze_proj{proj}_N{N:05.2f}_Rb{Rb:03.0f}_nx{nx:04d}.ipynb"
    )
    path_pdf = path_ipynb.with_suffix(".pdf")

    date_in = modification_date(path_in)
    try:
        date_out = modification_date(path_out)
    except FileNotFoundError:
        has_to_run = True
    else:
        has_to_run = date_in > date_out

    if has_to_run:
        pm.execute_notebook(
            path_in, path_out, parameters=dict(path_dir=str(path))
        )
        print(f"{path_out} saved")

    date_in = modification_date(path_ipynb)
    try:
        date_out = modification_date(path_pdf)
    except FileNotFoundError:
        has_to_run = True
    else:
        has_to_run = date_in > date_out

    if has_to_run:
        os.system(f"jupyter-nbconvert --to pdf {path_ipynb}")
