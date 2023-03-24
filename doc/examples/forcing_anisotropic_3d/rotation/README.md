# Study on inertial wave turbulence forced by large scales and large frequencies

Grid Ro and Re.

Standard viscosity (to get resolve the Kolmogorov scale):

```
nu ~ injection_rate ** (1 / 3) / (k_max / coef) ** (4 / 3)
```

The `Ro` and resolution `n` are defined in `util.py`, and `Re` is fixed according to the resolution using `coef = 2`. The aspect ratio is fixed to one in all the simulations. 

All the simulations are runned on the cluster Jean-Zay. 

## n = 320 (t_statio = 100)

We start by small simulations for `n = 320`. 
Each simulation is performed using 4 nodes (160 cores) and `fftw3d`.

| Ro           | 1.000e-00     | 1.000e-01     | 3.162e-02     | 1.000e-02     | 
|--------------|---------------|---------------|---------------|---------------|
| time fft (s) |               |               |               |               |              
| time / simul | ? days        | ? days        | ? days        | ? days        | 
| h.CPU        |               |               |               |               |            

## n = 640 (t_statio = 120)

We restart from the final state of the simulations with `n = 320`. We double the resolution in all direction (8 times more grid points) and the viscosity is decreased by `2^(4/3)`.
Each simulation is performed using 16 nodes (640 cores) and `pfft`.

| Ro           | 1.000e-00     | 1.000e-01     | 3.162e-02     | 1.000e-02     | 
|--------------|---------------|---------------|---------------|---------------|
| time fft (s) |               |               |               |               |              
| time / simul | ? days        | ? days        | ? days        | ? days        | 
| h.CPU        |               |               |               |               |  

## nh = 1280 (t_statio = 130)

We restart from the final state of the simulations with `n = 640`. We double the resolution in all direction (8 times more grid points) and the viscosity is decreased by `2^(4/3)`.
Each simulation is performed using 64 nodes (2560 cores) and `pfft`.

| Ro           | 1.000e-00     | 1.000e-01     | 3.162e-02     | 1.000e-02     | 
|--------------|---------------|---------------|---------------|---------------|
| time fft (s) |               |               |               |               |              
| time / simul | ? days        | ? days        | ? days        | ? days        | 
| h.CPU        |               |               |               |               |  





