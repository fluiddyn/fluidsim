from util import couples320, get_path_finer_resol, get_customized_dataframe

paths = []
for N, Rb in sorted(couples320):
    paths.append(get_path_finer_resol(N, Rb))

print(f"Using {len(paths)} simulations")

df = get_customized_dataframe(paths)
