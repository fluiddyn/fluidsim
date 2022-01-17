import argparse
from pathlib import Path

from fluiddyn.util import mpi
from fluidsim import load_for_restart

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str)

args = parser.parse_args()
mpi.printby0(args)

path_dir = Path(args.path)

if not path_dir.exists():
    raise ValueError

params, Simul = load_for_restart(path_dir)

sim = Simul(params)

period = sim.forcing.get_info()["period"]
sim.params.time_stepping.t_end = 4 * period

sim.time_stepping.start()
