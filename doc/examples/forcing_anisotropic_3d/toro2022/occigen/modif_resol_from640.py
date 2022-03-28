import os
from pathlib import Path
import subprocess

path_scratch = Path(os.environ["SCRATCHDIR"])
path_init = path_scratch / "2022/aniso/init_occigen"

paths = sorted(path_init.glob("ns3d.strat_toro*_640x640*"))

for path in paths:
    try:
        new_path = next(path.glob("State_phys_896x896*/state_phys_*.h5"))
    except StopIteration:
        subprocess.run(
            f"fluidsim-modif-resolution {path} 14/10".split(), check=True
        )
    else:
        print(f"{new_path} already created")
