"""
Needs some memory:

```
oarsub -I -l "{cluster='calcul2'}/nodes=1/core=4"
conda activate env_fluidsim
python postrun320.py
```

For each finished simulation:

1. clean up the directory
2. if the corresponding simulation for nh=640 is not present, prepare a file for nh=640.
   else, delete this file.
4. execute and save a notebook analyzing the simulation
"""

from pathlib import Path
from shutil import rmtree
import re
import subprocess

import papermill as pm

from fluiddyn.util import modification_date
from fluidsim.util import times_start_last_from_path, load_params_simul

nh = 320
t_end = 20.0

deltat = 0.1

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

path_output_papermill = path_base / "results_papermill"
path_output_papermill.mkdir(exist_ok=True)

paths = sorted(path_base.glob("aniso/ns3d*_toro_*_320x320x*"))
paths640 = sorted(path_base.glob("aniso/ns3d*_toro_*_640x640x*"))

couples_no_640 = (10, 5)

for path in paths:
    t_start, t_last = times_start_last_from_path(path)

    if t_last < t_end:
        print(f"{path.name:90s} not finished ({t_last=})")
        continue
    print(f"{path.name:90s} done ({t_last=})")

    # delete some useless restart files
    params = load_params_simul(path)
    deltat_file = params.output.periods_save.phys_fields
    path_files = sorted(path.glob("state_phys*"))
    for path_file in path_files:
        time = float(path_file.name.rsplit("_t", 1)[1][:-3])
        if time % deltat_file > deltat:
            print(f"deleting {path_file.name}")
            path_file.unlink()

    N = float(params.N)
    nx = params.oper.nx

    N_str = "_N" + re.search(r"_N(.*?)_", path.name).group(1)

    Rb_str = re.search(r"_Rb(.*?)_", path.name).group(1)
    Rb = float(Rb_str)
    Rb_str = "_Rb" + Rb_str

    try:
        corresponding_path640 = next(
            p for p in paths640 if N_str in p.name and Rb_str in p.name
        )
    except StopIteration:
        corresponding_path640 = False

    try:
        path_init_file = next(path.glob("State_phys_640x640*/state_phys_t*.h5"))
    except StopIteration:
        path_init_file = False

    if not corresponding_path640 and (N, Rb) not in couples_no_640:
        if not path_init_file:
            subprocess.run(
                ["fluidsim-modif-resolution", str(path), "2"],
                check=True,
            )
    else:
        if path_init_file:
            path_init_dir = path_init_file.parent
            print(f"deleting {path_init_dir}")
            rmtree(path_init_dir, ignore_errors=True)

    path_in = "analyse_1simul_papermill_short.ipynb"
    path_ipynb = path_out = (
        path_output_papermill
        / f"analyze_N{N:05.2f}_Rb{Rb:03.0f}_nx{nx:04d}.ipynb"
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
        p = subprocess.run(
            f"jupyter-nbconvert --to pdf {path_ipynb}".split(), check=True
        )
