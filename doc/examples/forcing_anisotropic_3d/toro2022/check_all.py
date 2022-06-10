from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from util import path_base, couples320, couples

t_ends = {320: 20, 640: 30, 896: 40, 1344: 44, 1792: 48, 2240: 50}

path_base_occigen = path_base / "from_occigen/aniso"
paths_all = sorted(
    list(path_base.glob("aniso/ns3d.strat*"))
    + list(path_base_occigen.glob("ns3d.strat*"))
)


for couple in sorted(couples320):
    N, Rb = couple
    str_N = f"_N{N}_"
    str_Rb = f"_Rb{Rb:.3g}_"
    paths_couple = [p for p in paths_all if str_N in p.name and str_Rb in p.name]
    paths_couple.sort(key=lambda p: int(p.name.split("x")[1]))
    print("--------")
    for nh, t_end in t_ends.items():
        if couple not in couples[nh]:
            break

        params = f"{N=} {Rb=} {nh=}"

        try:
            path = next(p for p in paths_couple if f"_{nh}x{nh}x" in p.name)
        except StopIteration:
            print(f"{params:40s}: not started")
            continue

        t_start, t_last = times_start_last_from_path(path)
        if t_last >= t_end:
            print(f"{params:40s}: completed")
        else:
            try:
                estimated_remaining_duration = (
                    get_last_estimated_remaining_duration(path)
                )
            except RuntimeError:
                estimated_remaining_duration = "?"
            print(
                f"{params:40s}: {t_last=} < {t_end} ({estimated_remaining_duration = })"
            )
            print(path)
