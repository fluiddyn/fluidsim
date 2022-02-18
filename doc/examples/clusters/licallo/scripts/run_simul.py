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
        N=10,
        forced_field="polo",
        # Problem with the energy injection if the following parameters... To check
        #F=1.0,    
        #ratio_kfmin_kf=0.9,
        #ratio_kfmax_kf=3.5,
        #init_velo_max=2.0,
        #t_end=2.0,
        #nz=40,
    )
