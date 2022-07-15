import sys
import os

from pathlib import Path

import pandas as pd

from fluiddyn.clusters.tgcc import SKL as Cluster

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

if "CCCWORKDIR" in os.environ:
    path_base_tgcc = Path(os.environ["CCCWORKDIR"])

    cluster = Cluster()
    cluster.commands_setting_env = [
        "pcocc-rs run ccc-rhel8",
        "module switch dfldatadir/gen7567",
        "module load mpi python3/3.8.10",
        "source $HOME/myvenv_system-site-packages/bin/activate",
        "export FLUIDSIM_PATH=$CCCSCRATCHDIR",
    ]
else:
    path_base_tgcc = path_base
    cluster = None

path_init_tgcc = path_base_tgcc / "init_tgcc"

new_simuls = pd.read_csv("new_simulations20220613.csv")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Launch simulations")

    parser.add_argument("N", type=float)
    parser.add_argument("Rb", type=float)
    parser.add_argument("nh", type=int)

    return parser.parse_args()


def get_sim_info_from_args(args):
    sims = new_simuls
    sim = sims[(sims.N == args.N) & (sims.R_i == args.Rb) & (sims.nh == args.nh)]

    if len(sim) == 0:
        print(f"No simulation corresponding to {args.__dict__}")
        sys.exit(0)

    sim = sim.loc[0]
    resolution = sim.init.split("_", 2)[1]
    sim.nh_small, sim.nz_small = tuple(
        int(n) for n in resolution.split("x")[0::2]
    )

    return sim


def type_fft_from_nhnz(nh, nz):
    raise NotImplementedError


def nb_nodes_from_nhnz(nh, nz):
    raise NotImplementedError


def get_info_jobs():

    process = run(f"squeue -u {user}")
    lines = process.stdout.split("\n")[1:]
    jobs_id = [line.split()[0] for line in lines if line]

    jobs_name = {}
    jobs_runtime = {}
    for job_id in jobs_id:
        process = run(f"scontrol show job {job_id}")
        for line in process.stdout.split("\n"):
            line = line.strip()
            if line.startswith("JobId"):
                job_name = line.split(" JobName=")[1]
                jobs_name[job_id] = job_name
            elif line.startswith("RunTime"):
                jobs_runtime[job_id] = line.split("RunTime=")[1].split(" ")[0]

    return jobs_id, jobs_name, jobs_runtime


cluster = None
