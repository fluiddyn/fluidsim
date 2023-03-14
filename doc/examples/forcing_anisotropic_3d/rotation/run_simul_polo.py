#!/usr/bin/env python
"""

For help, run

```
./run_simul_polo.py -h
```

"""
from fluidsim.util.scripts.turb_rotation_trandom_anisotropic import main

if __name__ == "__main__":

    params, sim = main(f=10, coef_nu=1.2, t_end=10, n=32, NO_GEOSTROPHIC_MODES=False)
