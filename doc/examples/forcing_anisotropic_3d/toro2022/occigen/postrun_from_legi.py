"""
Needs a lot of RAM for simulations for nh=896 and large N (80 and 120):

```bash
ssh -X cl1f001
nice -10 python postrun_from_legi.py
```

"""

from pathlib import Path
import subprocess
import re

import papermill as pm

from fluiddyn.util import modification_date

from fluidsim import load
from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
    load_params_simul,
)

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS/from_occigen/aniso")

paths_largeN = [
    p
    for p in sorted(path_base.glob("ns3d*_N[81]*"))
    if (
        ("_N80_" in p.name or "_N120_" in p.name)
        and ("896x896" in p.name or "1344x1344" in p.name)
    )
]

for path in paths_largeN:
    t_start, t_last = times_start_last_from_path(path)

    if "896x896" in path.name:
        nh = 896
        t_end = 40.0
    elif "1344x1344" in path.name:
        nh = 1344
        t_end = 44.0

    if t_last < t_end:
        try:
            estimated_remaining_duration = get_last_estimated_remaining_duration(
                path
            )
        except RuntimeError:
            estimated_remaining_duration = "?"

        print(
            f"{path.name:90s} not finished ({t_last=}, {estimated_remaining_duration=})"
        )
        continue

    params = load_params_simul(path)
    N = float(params.N)
    Rb = float(re.search(r"_Rb(.*?)_", path.name).group(1))

    tmp = f"{N=} {Rb=} {nh=}"
    print(f"{tmp:40s}: completed")

    # compute spatiotemporal spectra
    sim = load(path, hide_stdout=True)
    t_statio = round(t_start) + 1.0
    if nh == 896 and N >= 80:
        t_statio += 4

    sim.output.spatiotemporal_spectra.get_spectra(tmin=t_statio)

    path_in = "../analyse_1simul_papermill.ipynb"
    path_output_papermill = path_base / "results_papermill"
    path_out = (
        path_output_papermill
        / f"analyze_N{N:05.2f}_Rb{Rb:03.0f}_nx{nh:04d}.ipynb"
    )

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

paths_ipynb = sorted(path_base.glob("results_papermill/analyze_*.ipynb"))

for path_ipynb in paths_ipynb:

    path_pdf = path_ipynb.with_suffix(".pdf")
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
