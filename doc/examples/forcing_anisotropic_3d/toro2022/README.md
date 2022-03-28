# Study on stratified turbulence forced by large scale columnar vortices

Grid N (related to Fh) and Rb (buoyancy Reynolds number).

Standard viscosity (to get a real Kolmogorov scale) and order 4 hyperviscosity:

```
nu_4 ~ injection_rate_4 ** (1 / 3) / k_max ** (10 / 3)
```

## nh = 320 (t_end = 20)

We start by `nh = 320` for:

```python
for N in [10, 20, 40]:
    for Rb in [5, 10, 20, 40, 80, 160]:
        if N == 40 and Rb == 160:
            continue
```

This corresponds to 17 simulations.

The aspect ratio is set by:

```python
def get_ratio_nh_nz(N):
    "Get the ratio nh/nz"
    if N == 40:
        return 8
    elif N == 20:
        return 4
    elif N == 10:
        return 2
    else:
        raise NotImplementedError
```

| N | 10          | 20         | 40         |
|---|-------------|------------|------------|
|   | 320x320x160 | 320x320x80 | 320x320x40 |

## nh = 640 (t_end = 30)

The resolution is doubled in all direction (8 times more grid points) and the hyperviscosity in decreased by 10 (`~2**(10/3)`).

| N | 10          | 20          | 40         |
|---|-------------|-------------|------------|
|   | 640x640x320 | 640x640x160 | 640x640x80 |

These simulations are done on a LEGI cluster (calcul8) on 10 cores. They last from ~2 day (480 h.CPU) to 10 days (2400 h.CPU).

## nh = 896 (t_end = 40)

The resolution is multiplied by 1.4 and the hyperviscosity in decreased by 3.07.

| N | 10          | 20          | 40          |
|---|-------------|-------------|-------------|
|   | 896x896x448 | 896x896x224 | 896x896x112 |

We just need 8 simulations for

- N = 10, Rb = 80 and 160
- N = 20, Rb = 20, 40, 80
- N = 40, Rb = 10, 20, 40

The simulations can be carried out on 4 nodes (28*4 = 112 cores) on the Occigen cluster.

## nh = 1280 (t_end = 44?)

The resolution is multiplied by 4/3 and the hyperviscosity in decreased by 2.61.

| N | 10            | 20            | 40            |
|---|---------------|---------------|---------------|
|   | 1280x1280x640 | 1280x1280x320 | 1280x1280x160 |

- N = 10, Rb = 160
- N = 20, Rb = 40, 80
- N = 40, Rb = 10, 20, 40

We should be able to run these simulations on 4 nodes (80 cores).

## nh = 1920 (t_end = 48?)

The resolution is multiplied by 3/2 and the hyperviscosity in decreased by 3.86.

| N | 20            | 40            |
|---|---------------|---------------|
|   | 1920x1920x480 | 1920x1920x240 |

- N = 20, Rb = 40, 80
- N = 40, Rb = 10, 20, 40

## nh = 2560 (t_end = 50?)

The resolution is multiplied by 4/3 and the hyperviscosity in decreased by 2.61.

| N | 20            | 40            |
|---|---------------|---------------|
|   | 2560x2560x480 | 2560x2560x320 |

- N = 20, Rb = 80
- N = 40, Rb = 20, 40
