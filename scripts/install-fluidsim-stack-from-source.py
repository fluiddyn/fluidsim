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


parser = argparse.ArgumentParser(prog=__file__, description="Fluidsim installer")

args = parser.parse_args()

def run_pip(command="install"):
    return subprocess.run([sys.executable, "-m", "pip", command])

run_pip("list")
