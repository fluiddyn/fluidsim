#!/usr/bin/env python
"""

For help, run

```
./run_simul_polo.py -h
```

"""
from fluidsim.util.scripts.turb_rotation_trandom_anisotropic import main

if __name__ == "__main__":

    params, sim = main(Ro=0.02, nu=0, coef_nu4=1, t_end=20, F=0.97, n=48, nkmin_forcing=3, nkmax_forcing=5)
