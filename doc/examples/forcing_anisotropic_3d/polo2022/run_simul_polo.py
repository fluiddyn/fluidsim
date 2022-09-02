#!/usr/bin/env python
"""

For help, run

```
./run_simul_polo.py -h
```

"""

from fluidsim.util.scripts.turb_trandom_anisotropic import main

if __name__ == "__main__":

    params, sim = main(
        forced_field="polo",
        F=0.3,
        ratio_kfmin_kf=0.5,
        ratio_kfmax_kf=2.0,
        init_velo_max=2.0,
        # delta_angle=0.2, # Replaced by delta_F
        t_end=100.0,
    )
