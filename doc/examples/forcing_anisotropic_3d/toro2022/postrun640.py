"""
Needs some memory:

```
oarsub -I -l "{cluster='calcul2'}/nodes=1/core=10"
conda activate env_fluidsim
python postrun640.py
```

For each finished simulation:

1. clean up the directory
2. prepare a directory with initial states for simulations on Occigen

This directory can be synchronized on Occigen with:

```
rsync -rvz /fsnet/project/meige/2022/22STRATURBANIS/init_occigen augier@occigen.cines.fr:/scratch/cnt0022/egi2153/augier/2022/aniso
```

3. compute the spatiotemporal spectra

4. execute and save a notebook analyzing the simulation

"""

from pathlib import Path
from shutil import copyfile, rmtree
import re
import subprocess
from itertools import product

import papermill as pm

from fluiddyn.util import modification_date
from fluidsim.util import times_start_last_from_path, load_params_simul
from fluidsim import load

t_end = 30.0
t_statio = 21.0

deltat = 0.1

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

path_output_papermill = path_base / "results_papermill"
path_output_papermill.mkdir(exist_ok=True)

path_init_occigen = path_base / "init_occigen"
path_init_occigen.mkdir(exist_ok=True)

paths = sorted(path_base.glob("aniso/ns3d*_toro_*_640x640x*"))


def lprod(a, b):
    return list(product(a, b))


couples_for_occigen = set(
    lprod([10], [80, 160])
    + lprod([20], [20, 40, 80])
    + lprod([40], [10, 20, 40, 80])
    + [(80, 10), (120, 10)]
)


for path in paths:
    t_start, t_last = times_start_last_from_path(path)

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
        if time % deltat_file > deltat:
            print(f"deleting {path_file.name}")
            path_file.unlink()

    # compute spatiotemporal spectra
    sim = load(path, hide_stdout=True)
    sim.output.spatiotemporal_spectra.get_spectra(tmin=t_statio)

    N = float(sim.params.N)
    nx = sim.params.oper.nx
    Rb = float(re.search(r"_Rb(.*?)_", path.name).group(1))

    path_init = path_init_occigen / path.name
    if (N, Rb) in couples_for_occigen:
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
