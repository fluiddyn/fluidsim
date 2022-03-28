import os
from pathlib import Path

from fluiddyn.clusters.cines import Occigen

here = Path(__file__).parent.absolute()

USER = os.environ.get("USER")

with open(here.parent / "setup_env_base.sh") as file:
    code_setup_env = file.readlines()

code_setup_env = [line.strip() for line in code_setup_env if line.strip()]
code_setup_env.append("conda activate env_" + USER)

# code = "\n".join(code_setup_env)
# print(code)

# sbatch: error: You asked for HSW24 nodes which are no more available on
# OCCIGEN since (3/1/2022). Ask BDW28 instead and eventually adapt you scripts to
# submit new jobs. WARNING : no node exceed 64 GB now on the machine.

Occigen.constraint = "BDW28"
Occigen.nb_cores_per_node = 28

try:
    cluster = Occigen()
except ValueError as error:
    print("warning: " + str(error))
    cluster = False
else:
    cluster.commands_setting_env = code_setup_env
