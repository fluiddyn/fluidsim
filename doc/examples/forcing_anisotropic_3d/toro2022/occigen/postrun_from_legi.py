from pathlib import Path
import subprocess

from fluiddyn.util import modification_date

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS/from_occigen/aniso")

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
