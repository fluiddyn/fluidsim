"""
For each finished simulation:

1. clean up the directory

2. compute the spatiotemporal spectra

3. compress spectra

4. save simulations folders in $WORK and $STORE on Jean-Zay
"""

from subprocess import run
import os

from fluidsim.util import times_start_last_from_path, load_params_simul
from fluidsim import load

from util import (
    get_t_end,
    get_t_statio,
    Ro_target,
    list_paths
)

def clean_sim_dir(path, DELETE_SPATIOTEMPORAL_SPECTRA=False):
    print("Cleaning simulation directory: ", path)
    t_start, t_last = times_start_last_from_path(path)
    params = load_params_simul(path)
    n = params.oper.nx
    t_end = get_t_end(n)
    t_statio = get_t_statio(n)

    # delete some useless restart files
    deltat = params.output.periods_save.phys_fields
    path_files = sorted(path.glob(f"state_phys*"))
    for path_file in path_files:
        time = float(path_file.name.rsplit("_t", 1)[1][:-3])
        if (
            # time % deltat_file > deltat
            time != t_last
            and abs(time - t_statio) > deltat
            and abs(time - t_end) > deltat
        ):
            print(f"deleting {path_file.name}")
            path_file.unlink()

    # delete spatiotemporal spectra
    if DELETE_SPATIOTEMPORAL_SPECTRA:
        path_files = sorted(path.glob(f"spatiotemporal/periodogram_*"))
        for path_file in path_files:
            print(f"deleting {path_file.name}")
            #path_file.unlink()


def compute_spatiotemporal_spectra(path):
    # compute spatiotemporal spectra
    params = load_params_simul(path)
    n = params.oper.nx
    t_statio = get_t_statio(n)
    sim = load(path, hide_stdout=True)
    sim.output.spatiotemporal_spectra.get_spectra(tmin=t_statio)
   


ns = [320, 640]

for n in ns:
    for NO_GEOSTROPHIC_MODES in [False, True]:
        for Ro in sorted(Ro_target):
            print("--------------------------------------------")
            path_runs = list_paths(Ro, n, NO_GEOSTROPHIC_MODES=NO_GEOSTROPHIC_MODES)
            assert len(path_runs) == 1
            path_sim = path_runs[0]
            print(path)
            t_start, t_last = times_start_last_from_path(path_sim)
            t_end = get_t_end(n)
            if t_last < t_end - 0.01:
                print(f"{path.name:90s} not finished ({t_last=})")
                return
            print(f"{path.name:90s} done ({t_last=})")

            # We remove useless files
            clean_sim_dir(path_sim)

            # Compute spatiotemporal spectra
            compute_spatiotemporal_spectra(path_sim)

            # Compress spectra
            names = ("spect_energy_budg.h5", "spectra_kzkh.h5")
            base_command = "h5repack -f SHUF -f GZIP=4"
            for name in names:
                path = path_sim / name
                path_uncompressed = path.with_stem(path.stem + "_uncompressed")
                if path_uncompressed.exists():
                    continue
                path_compressed = path.with_stem(path.stem + "_compressed")
                command = f"{base_command} {path.name} {path_compressed.name}"
                run(command.split(), check=True, cwd=path_sim)
                path.rename(path_uncompressed)
                path_compressed.rename(path)

            # Save simul
            os.system(
                f"rsync -rvz -L --update {path_sim} /gpfswork/rech/uzc/uey73qw/aniso_rotation/"
            )

os.system(
    f"rsync -rvz -L --update /gpfsscratch/rech/uzc/uey73qw/aniso_rotation/ /gpfsstore/rech/uzc/uey73qw/aniso_rotation/"
)