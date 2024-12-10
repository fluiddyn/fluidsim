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

args = parser.parse_args()
print(args)

capture_output_default = not args.verbose


names_wheel = {"pyfftw": "pyFFTW"}


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
    uninstall=args.uninstall,
):
    requirement_specifier = words[0]

    if "@" in requirement_specifier:
        name_package = requirement_specifier.split("@")[0]
    else:
        name_package = requirement_specifier

    if uninstall:
        run_pip("uninstall", name_package, "--yes", check=False)

    command = ["install", *words]
    if rebuild:
        proc_ = run_pip("list", capture_output=True, echo=False)
        packages = [
            line.split()[0] for line in proc_.stdout.split("\n")[2:] if line
        ]
        if name_package not in packages:
            name_wheel = names_wheel.get(
                name_package, name_package.replace("-", "_")
            )
            run_pip("cache", "remove", name_wheel)
        command.extend(["--no-binary", name_package])

    if native:
        command.extend(["--config-settings", "setup-args=-Dnative=true"])

    return run_pip(*command, env=env)


proc = run_pip("list", capture_output=True)
lines = [
    line
    for line in proc.stdout.split("\n")[2:]
    if line and not any(line.startswith(name) for name in ["pip", "setuptools"])
]
if lines:
    warnings.warn(f"Virtual env is not clean. Packages installed:\n{proc.stdout}")


pip_install("mpi4py", rebuild=True)

# TODO: tempdir and requirements.txt

pip_install("pyfftw", rebuild=True)
pip_install("fluidfft", rebuild=True, native=not args.no_native)

pip_install("fluidfft-fftw", rebuild=True)

pip_install("fluidfft-fftwmpi", rebuild=True)
pip_install("fluidfft-mpi_with_fftw", rebuild=True)

pip_install("fluidsim", rebuild=True, native=not args.no_native)

pip_install("pytest", "pytest-mpi", "pytest-allclose", "pytest-mock", "ipython")

# with Python 3.13 and h5py<=3.12.1 we need (see https://github.com/h5py/h5py/issues/2523)
# pip cache remove h5py; HDF5_MPI="ON" CC=mpicc pip install h5py@git+https://github.com/h5py/h5py --no-binary h5py
if not args.no_h5py:
    env_hdf5_build = os.environ.copy()
    env_hdf5_build.update({"HDF5_MPI": "ON", "CC": "mpicc"})

    package_name = "h5py"
    if sys.version_info[:2] >= (3, 13):
        package_name += "@git+https://github.com/h5py/h5py"
    pip_install(package_name, rebuild=True, env=env_hdf5_build, uninstall=True)

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
mpirun -np 2 python -c 'import h5py; h5py.run_tests()'
pytest --pyargs fluidsim
mpirun -np 2 pytest --pyargs fluidsim
"""
)
