import sys

from pathlib import Path

import pandas as pd

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

path_base_tgcc = path_base
# path_base_tgcc = Path("/tmp")

path_init_tgcc = path_base_tgcc / "init_tgcc"

new_simuls = pd.read_csv("new_simulations20220613.csv")

# cluster.commands_setting_env = [
#     "source /etc/profile",
#     ". $HOME/miniconda3/etc/profile.d/conda.sh",
#     "conda activate env_fluidsim",
#     f"export FLUIDSIM_PATH={path_base}",
# ]


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


cluster = None
