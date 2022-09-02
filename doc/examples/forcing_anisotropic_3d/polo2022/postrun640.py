"""
Needs some memory:

```
srun -t 2:0:0 -p short hostname
conda activate env_fluidsim
python postrun640.py
```

For each finished simulation:

1. clean up the directory
2. prepare a directory with initial states for simulations on Jean-Zay

This directory can be synchronized on Jean-Zay with:

```
du -h /scratch/vlabarre/aniso/init_jeanzay
rsync -rvz -L --update /scratch/vlabarre/aniso/init_jeanzay/ns3d* uey73qw@jean-zay.idris.fr:/gpfsscratch/rech/uzc/uey73qw/aniso 
```

3. compute the spatiotemporal spectra

4. execute and save a notebook analyzing the simulation

5. send end_states to Jean-Zay
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
    path_base,
    path_output_papermill,
    get_t_end,
    get_t_statio,
    couples640,
    couples1280,
)

nh = 640
deltat = 0.1

path_output_papermill.mkdir(exist_ok=True)

path_end_states = path_base / "end_states"
path_end_states.mkdir(exist_ok=True)

path_init_jeanzay = path_base / "init_jeanzay"
path_init_jeanzay.mkdir(exist_ok=True)

paths = sorted(path_base.glob("ns3d*_polo_*_640x640x*"))

couples_for_jeanzay = couples1280

for path in paths:
    print(path)
    t_start, t_last = times_start_last_from_path(path)

    params = load_params_simul(path)
    N = float(params.N)
    nx = params.oper.nx
    proj = params.projection
    t_end = get_t_end(N, nh)
    t_statio = get_t_statio(N, nh)

    # Simulations with nu = 0 where just for testing on Licallo
    if params.nu_2 == 0.0:
        print(f"{path.name:90s} corresponds to a simulation with nul viscosity)")
        continue

    if t_last < t_end - 0.01:
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
    Rb = float(re.search(r"_Rb(.*?)_", path.name).group(1))

    path_init = path_init_jeanzay / path.name
    if (N, Rb) in couples_for_jeanzay:
        path_init.mkdir(exist_ok=True)

        path_to_copy = next(path.glob(f"state_phys_t00{t_end}*"))
        path_new = path_init / path_to_copy.name

        if not path_new.exists():
            print(f"copying in {path_new}")
            copyfile(path_to_copy, path_new)
    else:
        if path_init.exists():
            print(f"deleting {path_init}")
            rmtree(path_init, ignore_errors=True)

    path_in = "analyse_1simul_papermill.ipynb"
    path_ipynb = path_out = (
        path_output_papermill
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

os.system(
    "rsync -rvz -L --update /scratch/vlabarre/aniso/init_jeanzay/ns3d* uey73qw@jean-zay.idris.fr:/gpfsscratch/rech/uzc/uey73qw/aniso"
)
