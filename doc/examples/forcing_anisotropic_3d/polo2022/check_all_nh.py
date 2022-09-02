from doc.examples.forcing_anisotropic_3d.polo2022.util import get_t_end
from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from util import path_base, path_base_jeanzay, couples, get_t_end

from shutil import copyfile, rmtree

nh = 1280

print(couples[nh])
print(len(couples[nh]), "x2 simulations to launch")

paths_all = sorted(list(path_base_jeanzay.glob(f"ns3d.strat_polo_*{nh}x{nh}*")))

for proj in ["None", "poloidal"]:
    for couple in sorted(couples[nh]):
        N, Rb = couple
        print(couple)
        str_N = f"_N{N}"
        str_Rb = f"_Rb{Rb:.3g}"
        if proj == "None":
            paths_couple = [
                p
                for p in paths_all
                if str_N in p.name and str_Rb in p.name and "proj" not in p.name
            ]
        elif proj == "poloidal":
            paths_couple = [
                p
                for p in paths_all
                if str_N in p.name and str_Rb in p.name and "proj" in p.name
            ]
        paths_couple.sort(key=lambda p: int(p.name.split("x")[1]))
        print("--------")
        t_end = get_t_end(N, nh)
        params = f"{proj=} {N=} {Rb=} {nh=}"

        try:
            path = next(p for p in paths_couple if f"_{nh}x{nh}x" in p.name)
            print(path)
        except StopIteration:
            print(f"{params:40s}: not started")
            continue

        try:
            path_init_file = next(path.glob(f"state_phys_t*.h5"))
        except StopIteration:
            print(f"No state_phys_t*.h5 in the directory, we remove it")
            rmtree(path, ignore_errors=True)
            continue

        t_start, t_last = times_start_last_from_path(path)

        if t_last >= t_end - 0.01:
            print(f"{params:40s}: completed: {t_last=} >= {t_end=} - 0.01")
        else:
            try:
                estimated_remaining_duration = (
                    get_last_estimated_remaining_duration(path)
                )
            except RuntimeError:
                estimated_remaining_duration = "?"
            print(
                f"{params:40s}: not finished: {t_last=} < {t_end=} ({estimated_remaining_duration=})"
            )
            print(path)
