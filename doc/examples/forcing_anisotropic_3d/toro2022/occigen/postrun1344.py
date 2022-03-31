from postrun896 import postrun, lprod


postrun(
    t_end=44.0,
    nh=1344,
    coef_modif_resol="4/3",
    couples_larger_resolution=set(
        lprod([20], [40, 80]) + lprod([40], [10, 20, 40])
    ),
)
