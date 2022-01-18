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

try:
    cluster = Occigen()
except ValueError as error:
    print("warning: " + str(error))
    cluster = False
else:
    cluster.commands_setting_env = code_setup_env
