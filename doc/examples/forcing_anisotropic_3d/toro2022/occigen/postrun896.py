"""
For each finished simulation:

1. clean up the directory
2. prepare a file with larger resolution
3. compute the spatiotemporal spectra
4. execute and save a notebook analyzing the simulation

"""

import re
import subprocess

import papermill as pm

from fluiddyn.util import modification_date
from fluidsim.util import times_start_last_from_path, load_params_simul
from fluidsim import load

from util import lprod, path_scratch


def postrun(t_end, nh, coef_modif_resol, couples_larger_resolution):

    nh_larger = int(nh * eval(coef_modif_resol))

    deltat = 0.05

    path_base = path_scratch / "aniso"

    path_output_papermill = path_base / "results_papermill"
    path_output_papermill.mkdir(exist_ok=True)

    paths = sorted(path_base.glob(f"ns3d*_toro_*_{nh}x{nh}x*"))

    for path in paths:
        t_start, t_last = times_start_last_from_path(path)

        if t_last < t_end:
            continue
        print(f"{path.name} done ({t_last=})")

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
        t_statio = round(t_start) + 1.0
        sim.output.spatiotemporal_spectra.get_spectra(tmin=t_statio)

        N = float(params.N)
        nx = params.oper.nx
        Rb = float(re.search(r"_Rb(.*?)_", path.name).group(1))

        try:
            next(path.glob(f"State_phys_{nh_larger}x{nh_larger}*"))
        except StopIteration:
            if (N, Rb) in couples_larger_resolution:
                subprocess.run(
                    f"fluidsim-modif-resolution {path} {coef_modif_resol}".split()
                )

        path_in = "../analyse_1simul_papermill.ipynb"
        path_out = (
            path_output_papermill
            / f"analyze_N{N:05.2f}_Rb{Rb:03.0f}_nx{nx:04d}.ipynb"
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


if __name__ == "__main__":
    postrun(
        t_end=40.0,
        nh=896,
        coef_modif_resol="3/2",
        couples_larger_resolution=set(
            lprod([10], [160])
            + lprod([20], [40, 80])
            + lprod([40], [10, 20, 40, 80])
        ),
    )
