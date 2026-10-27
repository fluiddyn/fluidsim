import os
from util import list_paths

path_dir = list_paths(N=60, Rb=10, nh=320, nz=80, proj="poloidal")

if len(path_dir) == 0:
    print("No simulation directory.")
elif len(path_dir) == 1:
    print(f"We move {path_dir[0]} on the cluster azzurra")
    os.system(
        f"rsync -rvz -L --progress {path_dir[0]} vlabarre@login-hpc.univ-cotedazur.fr:/workspace/vlabarre/aniso/"
    )
else:
    print("More than one simulation directory. Nothing is done.")
