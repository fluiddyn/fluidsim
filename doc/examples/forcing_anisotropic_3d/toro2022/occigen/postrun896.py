"""
For each finished simulation:

1. clean up the directory
2. prepare a file with larger resolution
3. compute the spatiotemporal spectra
4. execute and save a notebook analyzing the simulation

"""

from util import postrun, couples1344


postrun(
    t_end=40.0,
    nh=896,
    coef_modif_resol="3/2",
    couples_larger_resolution=couples1344,
)
