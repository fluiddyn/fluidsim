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

| N            | 10            | 20            | 40             |
|--------------|---------------|---------------|----------------|
|              | 896x896x448   | 896x896x224   | 896x896x112    |
| Rb           | 80, 160       | 20, 40, 80    | 10, 20, 40, 80 |
| # nodes      | 4             | 4             | 4              |
| # cores      | 112           | 112           | 112            |
| fft lib      | fftw1d        | fftw1d        | fftw1d         |
| time fft (s) | 0.19          | 0.11          | 0.06           |
| time / simul | 5 days        | 2.5 days      | 1.25 day       |
| h.CPU        | 27000         | 20000         | 13000          |

This should cost something like 22.5 days * 112 cores = 60_000 h.CPU

## nh = 1344 (t_end = 44?)

The resolution is multiplied by 3/2 and the hyperviscosity in decreased by 3.86.

| N            | 10            | 20            | 40            |
|--------------|---------------|---------------|---------------|
|              | 1344x1344x672 | 1344x1344x336 | 1344x1344x168 |
| Rb           | 160           | 40, 80        | 10, 20, 40    |
| # nodes      | 8             | 4             | 8             |
| # cores      | 224           | 112           | 224           |
| fft lib      | fftw1d        | fftw1d        | p3dfft        |
| time fft (s) | 0.33          | 0.31          | 0.11          |

## nh = 1792 (t_end = 48?)

The resolution is multiplied by 4/3 and the hyperviscosity in decreased by 2.61.

| N            | 20            | 40            |
|--------------|---------------|---------------|
|              | 1792x1792x448 | 1792x1792x224 |
| Rb           | 40, 80        | 10, 20, 40    |
| # nodes      | 16            | 16            |
| fft lib      | fftw1d        | p3dfft        |
| time fft (s) | 0.30          | 0.16          |


## nh = 2240 (t_end = 50?)

The resolution is multiplied by 4/3 and the hyperviscosity in decreased by 2.61.

| N       | 20            | 40            |
|---------|---------------|---------------|
|         | 2240x2240x560 | 2240x2240x280 |
| Rb      | 80            | 20, 40        |
| # nodes | 16            | 16            |
| fft lib | ?             | ?             |
