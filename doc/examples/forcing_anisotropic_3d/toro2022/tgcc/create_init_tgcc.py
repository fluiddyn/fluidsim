from util import new_simuls, path_base

path_init_tgcc = path_base / "init_tgcc"

path_end_state_occigen = path_base / "from_occigen/end_states"
path_legi = path_base / "aniso"

for name in new_simuls.init:

    if name.startswith("("):
        continue
    nh = int(name.split("_", 1)[1].split("x", 1)[0])

    if nh > 640:
        path_base_input = path_end_state_occigen
    else:
        path_base_input = path_legi

    path_input = sorted(path_base_input.glob(f"*{name}*/state_phys_*"))[-1]
    print(path_input)

    link_last_state = path_init_tgcc / path_input.parent.name / path_input.name

    if not link_last_state.exists():
        link_last_state.parent.mkdir(exist_ok=True)
        link_last_state.symlink_to(path_input)
        print(f"Link {link_last_state} created")
