#!/usr/bin/env python
"""

For help, run

```
./run_simul.py -h
```

"""

from fluidsim.util.scripts.turb_trandom_anisotropic import main


if __name__ == "__main__":
    params, sim = main(
        forced_field="polo",
        F=1.0,
        ratio_kfmin_kf=0.9,
        ratio_kfmax_kf=1.5,
        init_velo_max=2.0,
        t_end=2.0,
        nz=40,
    )
