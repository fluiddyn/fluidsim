# Study on stratified turbulence forced by large scale columnar vortices

Grid N (related to Fh) and Rb (buoyancy Reynolds number).

Standard viscosity (to get a real Kolmogorov scale) and order 4 hyperviscosity:

```
nu_4 ~ injection_rate_4 ** (1 / 3) / k_max ** (10 / 3)
```

## nh = 320 (t_end = 20)

We start by small simulations for `nh = 320`. The couples `N, Rb` are defined
in `util.py` (and `occigen/util.py`).

The aspect ratio is set by something like (see `util.py`):

```python
def get_ratio_nh_nz(N):
    "Get the ratio nh/nz"
    if N >= 80:
        return 16
    elif N == 40:
        return 8
    elif N in [20, 30]:
        return 4
    elif N <= 15:
        return 2
    else:
        raise NotImplementedError
```

| N | <10         | 20, 30     | 40         | 80 & 120   |
|---|-------------|------------|------------|------------|
|   | 320x320x160 | 320x320x80 | 320x320x40 | 320x320x20 |

## nh = 640 (t_end = 30)

The resolution is doubled in all direction (8 times more grid points) and the
hyperviscosity is decreased by 10 (`~2**(10/3)`).

| N | <10         | 20, 30      | 40         | 80 & 120   |
|---|-------------|-------------|------------|------------|
|   | 640x640x320 | 640x640x160 | 640x640x80 | 640x640x40 |

These simulations are done on a LEGI cluster (calcul8) on 20 cores.

## nh = 896 (t_end = 40)

The resolution is multiplied by 1.4 and the hyperviscosity is decreased by 3.07.

| N            | 10            | 20            | 40             | 80 & 120   |
|--------------|---------------|---------------|----------------|------------|
|              | 896x896x448   | 896x896x224   | 896x896x112    | 896x896x56 |
| Rb           | 80, 160       | 20, 40, 80    | 10, 20, 40, 80 | 10         |
| # nodes      | 4             | 4             | 4              | 2          |
| # cores      | 112           | 112           | 112            | 56         |
| fft lib      | fftw1d        | fftw1d        | fftw1d         | fftwmpi3d  |
| time fft (s) | 0.19          | 0.11          | 0.06           | 0.042      |
| time / simul | 5 days        | 2.5 days      | 1.25 day       | ?          |
| h.CPU        | 27000         | 20000         | 13000          |            |

This should cost something like 22.5 days * 112 cores = 60_000 h.CPU.

## nh = 1344 (t_end = 44)

The resolution is multiplied by 3/2 and the hyperviscosity is decreased by 3.86.

| N            | 10            | 20            | 40            | 80 & 120     |
|--------------|---------------|---------------|---------------|--------------|
|              | 1344x1344x672 | 1344x1344x336 | 1344x1344x168 | 1344x1344x84 |
| Rb           | 160           | 40, 80        | 10, 20, 40, 80| 10           |
| # nodes      | 12            | 8             | 8             | 4            |
| # cores      | 336           | 224           | 224           | 112          |
| fft lib      | fftw1d        | p3dfft        | p3dfft        | p3dfft       |
| time fft (s) | ?             | 0.25          | 0.11          | ?            |

## nh = 1792 (t_end = 48)

The resolution is multiplied by 4/3 and the hyperviscosity is decreased by 2.61.

| N            | 20            | 40            | 80 & 120      |
|--------------|---------------|---------------|---------------|
|              | 1792x1792x448 | 1792x1792x224 | 1792x1792x112 |
| Rb           | 40, 80        | 10, 20, 40    | 10            |
| # nodes      | 16            | 16            | 8             |
| # cores      | 448           | 448           | 224           |
| fft lib      | fftw1d        | p3dfft        | p3dfft        |
| time fft (s) | 0.30          | 0.16          | ?             |

## nh = 2240 (t_end = 50)

The resolution is multiplied by 5/4 and the hyperviscosity is decreased by 2.10.

| N       | 20            | 40            | 80 & 120      |
|---------|---------------|---------------|---------------|
|         | 2240x2240x560 | 2240x2240x280 | 2240x2240x140 |
| Rb      | 80            | 20, 40        | 10            |
| # nodes | 16            | 16            | 8             |
| fft lib | p3dfft        | p3dfft        | p3dfft        |

## Distribution of the data

### Links

#### How to

- https://www.datacc.org/vos-besoins/valoriser-ses-donnees/deposer-ses-donnees-en-ligne-ou-et-comment/
- https://creativecommons.org/share-your-work/
- https://creativecommons.org/about/cclicenses/

#### Database

- https://turbase.cineca.it
- http://turbulence.pha.jhu.edu/
- https://www.seanoe.org
- https://www.data-terra.org/nous-connaitre/presentation-de-data-terra/

#### Snapshot of the code

- https://doc.archives-ouvertes.fr/deposer/deposer-le-code-source/
- https://www.softwareheritage.org

### Setup for opening

1. Data for the papers with [UGA cloud](https://cloud.univ-grenoble-alpes.fr)
   (warning: limited to 50 Go)

2. HAL/Software Heritage for a DOI for Fluidsim 0.7.0 (or Zenodo?)

3. A dataset on a database ([Turbase](https://turbase.cineca.it) would be great)

   3 directories:

   - "Small" data for plot
   - 1 notebook per simulation (ipynb + html + pdf)
   - 1 state file per simulation (warning: big)

4. Dataset (selection < 50 Go) on Zenodo

5. Full data at LEGI (access?)

6. 1 paper describing the dataset and the regimes

7. Other papers based on this dataset (mixing, spatiotemporal, ...)
