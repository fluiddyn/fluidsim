from pathlib import Path

from subprocess import run

here = Path(__file__).absolute().parent

with open(here / "occigen/simuls_not_finished.txt", "r") as file:
    simuls_not_finished = [line.strip() for line in file.readlines()]

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")
paths = sorted(path_base.glob("aniso/ns3d*")) + sorted(
    path_base.glob("from_occigen/aniso/ns3d*")
)

names = ("spect_energy_budg.h5", "spectra_kzkh.h5")

base_command = "h5repack -f SHUF -f GZIP=4"

for path_sim in paths:

    if path_sim.name in simuls_not_finished:
        continue

    print(path_sim)
    for name in names:
        path = path_sim / name
        path_uncompressed = path.with_stem(path.stem + "_uncompressed")
        if path_uncompressed.exists():
            continue
        path_compressed = path.with_stem(path.stem + "_compressed")
        command = f"{base_command} {path.name} {path_compressed.name}"
        run(command.split(), check=True, cwd=path_sim)
        path.rename(path_uncompressed)
        path_compressed.rename(path)
