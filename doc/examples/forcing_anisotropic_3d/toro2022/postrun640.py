"""
Needs some memory:

```
oarsub -I -l "{cluster='calcul2'}/nodes=1/core=10"
conda activate env_fluidsim
python postrun640.py
```

For each finished simulations:

- clean up the directory
- prepare a directory with initial states for simulations on Occigen

This directory can be synchronized on Occigen with:

```
rsync -rvz /fsnet/project/meige/2022/22STRATURBANIS/init_occigen augier@occigen.cines.fr:/scratch/cnt0022/egi2153/augier/2022/aniso
```

- compute the spatiotemporal spectra

"""

from pathlib import Path
from shutil import copyfile

import papermill as pm

from fluidsim.util import times_start_last_from_path, load_params_simul
from fluidsim import load

t_end = 30.0
t_statio = 21.0

deltat = 0.1

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

path_init_occigen = path_base / "init_occigen"
path_init_occigen.mkdir(exist_ok=True)

paths = sorted(path_base.glob("aniso/ns3d*_toro_*_640x640x*"))


for path in paths:
    t_start, t_last = times_start_last_from_path(path)

    if t_last < t_end:
        continue
    print(f"{path.name} {t_last}")

    path_init = path_init_occigen / path.name
    path_init.mkdir(exist_ok=True)

    path_to_copy = next(path.glob(f"state_phys_t00{t_end}*"))
    path_new = path_init / path_to_copy.name

    if not path_new.exists():
        print(f"copying in {path_new}")
        copyfile(path_to_copy, path_new)

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
    sim = load(path)
    sim.output.spatiotemporal_spectra.get_spectra(tmin=t_statio)

    # TODO: better place, better name
    path_out = path / "analyze_1simul.ipynb"
    if not path_out.exists():
        pm.execute_notebook(
            "analyse_1simul.ipynb",
            path_out,
            parameters=dict(path_dir=str(path)),
        )
