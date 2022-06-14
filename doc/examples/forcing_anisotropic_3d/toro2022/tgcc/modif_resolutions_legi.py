import subprocess

from util import new_simuls, path_init_tgcc


for name, nh_after, nz_after in zip(
    new_simuls.init, new_simuls.nh, new_simuls.nz
):

    if name.startswith("("):
        continue

    resolution = name.split("_", 2)[1]
    print(resolution)

    nh_init, nz_init = tuple(int(n) for n in resolution.split("x")[0::2])

    path_simul_init = next(path_init_tgcc.glob(f"*{name}*"))

    print(path_simul_init)

    try:
        new_path = next(
            path_simul_init.glob(
                f"State_phys_{nh_after}x{nh_after}x{nz_after}/state_phys_*.h5"
            )
        )
    except StopIteration:
        command = (
            f"fluidsim-modif-resolution {path_simul_init} {nh_after}/{nh_init}"
        )
        print(command)
        # subprocess.run(command.split(), check=True)
    else:
        print(f"{new_path} already created")
