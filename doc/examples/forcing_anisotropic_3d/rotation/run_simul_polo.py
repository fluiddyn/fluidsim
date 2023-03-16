#!/usr/bin/env python
"""

For help, run

```
./run_simul_polo.py -h
```

"""
from fluidsim.util.scripts.turb_rotation_trandom_anisotropic import main

if __name__ == "__main__":

    params, sim = main(Ro=1e-1, coef_nu=2.0, t_end=2, n=32, NO_GEOSTROPHIC_MODES=False)
