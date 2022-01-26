#!/usr/bin/env python
"""

For help, run

```
./run_simul.py -h
```

"""
from fluidsim.util.scripts.turb_trandom_anisotropic import main

if __name__ == "__main__":

    params, sim = main(N=100, nz=160)
