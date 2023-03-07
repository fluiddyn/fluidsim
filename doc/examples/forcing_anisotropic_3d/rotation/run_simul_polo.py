#!/usr/bin/env python
"""

For help, run

```
./run_simul_polo.py -h
```

"""
from fluidsim.util.scripts.turb_rotation_trandom_anisotropic import main

if __name__ == "__main__":

    params, sim = main(f=1, coef_nu=1.0, coef_nu4=0, t_end=20)
