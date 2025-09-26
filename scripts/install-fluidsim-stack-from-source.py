#!/usr/bin/env python
"""Install fluidsim and few deps from source

## Download this script with wget or curl

```sh
rm -f install-fluidsim-stack-from-source.py
wget https://foss.heptapod.net/fluiddyn/fluidsim/-/raw/branch/default/scripts/install-fluidsim-stack-from-source.py
```

or

```sh
rm -f install-fluidsim-stack-from-source.py
curl -L -O https://foss.heptapod.net/fluiddyn/fluidsim/-/raw/branch/default/scripts/install-fluidsim-stack-from-source.py
```

## Usage

This script has to be used from a clean virtual environment (with pip installed).
The virtual environment can be created with different methods like

```sh
python3 -m venv venv-fluidsim
```

or (with miniforge):

```sh
conda create -n venv-fluidsim python pip
```

Then one needs to setup her-his environment by using few environment
variables, like `PATH` (for `mpicc`, `python` and `pip`), `CPATH`, `LIBRARY_PATH`,
`LD_LIBRARY_PATH` and `PKG_CONFIG_PATH`.

This can typically be done with modules, for example:

```sh
module load gcc openmpi fftw hdf5
```

Finally, launch the install script:

```sh
./install-fluidsim-stack-from-source.py
```

See `./install-fluidsim-stack-from-source.py -h` for options.

## Examples

On Adastra, one can run:

```sh
module purge
module load cpe/23.12
module load craype-x86-genoa
module load PrgEnv-gnu
module load gcc/13.2.0
module load cray-hdf5-parallel cray-fftw
module load cray-python

export LIBRARY_PATH=/opt/cray/pe/fftw/3.3.10.6/x86_genoa/lib
export CFLAGS="-I/opt/cray/pe/fftw/3.3.10.6/x86_genoa/include"
export PYFFTW_LIB_DIR="/opt/cray/pe/fftw/3.3.10.6/x86_genoa/lib"
export PYFFTW_INCLUDE="/opt/cray/pe/fftw/3.3.10.6/x86_genoa/include"

python -m venv ~/venv-fluidsim
. ~/venv-fluidsim/bin/activate
pip install pip -U
python ~/dev/fluidsim/scripts/install-fluidsim-stack-from-source.py --fftw-openmp -v
```

"""

import argparse
import subprocess
import sys
import warnings
import os


parser = argparse.ArgumentParser(prog=__file__, description="Fluidsim installer")

parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Give more output. Option is additive, and can be used up to 3 times.",
)

# TODO: "-r", "--requirements-file"

parser.add_argument("--uninstall", action="store_true")
parser.add_argument("--no-h5py", action="store_true")
parser.add_argument("--no-native", action="store_true")
parser.add_argument("--fftw-openmp", action="store_true")

args = parser.parse_args()
print(args)

capture_output_default = not args.verbose


def run_pip(
    *args, env=None, capture_output=capture_output_default, check=True, echo=True
):
    command = [sys.executable, "-m", "pip", *args]
    if echo:
        print(" ".join(command[2:]))
    return subprocess.run(
        command,
        check=check,
        text=True,
        env=env,
        capture_output=capture_output,
    )


def pip_install(
    *words,
    rebuild=False,
    native=False,
    env=None,
    env_update=None,
    uninstall=args.uninstall,
):
    requirement_specifier = words[0]

    name_package = requirement_specifier
    for char in "@[":
        if "@" in requirement_specifier:
            name_package = name_package.split(char)[0]

    if uninstall:
        run_pip("uninstall", name_package, "--yes", check=False)

    command = ["install", *words]
    if rebuild:
        proc_ = run_pip("list", capture_output=True, echo=False)
        packages = [
            line.split()[0] for line in proc_.stdout.split("\n")[2:] if line
        ]
        if name_package not in packages:
            name_wheel = name_package.replace("-", "_")
            run_pip("cache", "remove", name_wheel)
        command.extend(["--no-binary", name_package])

    if native:
        command.extend(["--config-settings", "setup-args=-Dnative=true"])

    if env_update is not None:
        assert env is None
        env = os.environ.copy()
        env.update(env_update)

    return run_pip(*command, env=env)


proc = run_pip("list", capture_output=True)
lines = [
    line
    for line in proc.stdout.split("\n")[2:]
    if line and not any(line.startswith(name) for name in ["pip", "setuptools"])
]
if lines:
    warnings.warn(f"Virtual env is not clean. Packages installed:\n{proc.stdout}")


pip_install("mpi4py", rebuild=True, env_update={"CFLAGS": "-O3"})

# TODO: tempdir and requirements.txt

# with Python 3.13 and h5py<=3.12.1 we need (see https://github.com/h5py/h5py/issues/2523)
# pip cache remove h5py; HDF5_MPI="ON" CC=mpicc pip install h5py@git+https://github.com/h5py/h5py --no-binary h5py
if not args.no_h5py:
    package_name = "h5py"
    if sys.version_info[:2] >= (3, 13):
        package_name += "@git+https://github.com/h5py/h5py"
    pip_install(
        package_name,
        rebuild=True,
        env_update={"HDF5_MPI": "ON", "CC": "mpicc"},
        uninstall=True,
    )

fftw_env = None
if args.fftw_openmp:
    fftw_env = {"CFLAGS": "-fopenmp"}
pip_install("pyfftw", rebuild=True, env_update=fftw_env)
pip_install("fluidfft", rebuild=True, native=not args.no_native)

pip_install("fluidfft-fftw", rebuild=True)

pip_install("fluidfft-fftwmpi", rebuild=True)
pip_install("fluidfft-mpi_with_fftw", rebuild=True)

pip_install(
    "fluidsim[test-mpi,pulp]@hg+https://foss.heptapod.net/fluiddyn/fluidsim",
    rebuild=True,
    native=not args.no_native,
)

proc = run_pip("freeze", capture_output=True)
name = "requirements-fluidsim-installer.txt"
with open(name, "w", encoding="utf-8") as file:
    file.write(proc.stdout)
print(f"requirements written in {name}")

subprocess.run(
    [
        sys.executable,
        "-c",
        "import h5py; print(h5py.version.info + f'mpi: {h5py.get_config().mpi}')",
    ],
    check=True,
)

print(
    """
You might want to run tests with commands like:
pytest --pyargs fluidsim
mpirun -np 2 python -c 'import h5py; h5py.run_tests()'
mpirun -np 2 pytest --pyargs fluidsim
"""
)
