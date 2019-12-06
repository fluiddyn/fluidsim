import sys

from fluidsim import load_for_restart

name_dir = sys.argv[-1]

if name_dir.endswith(".py"):
    print(
        "Use this script with something like\n"
        "python restart.py waves_coriolis/ns3d.strat_64x64x16_V4x4x1_2019-12-06_23-19-16"
    )
    sys.exit(1)

params, Simul = load_for_restart(name_dir)

params.time_stepping.t_end += 10.0

sim = Simul(params)

sim.time_stepping.start()
