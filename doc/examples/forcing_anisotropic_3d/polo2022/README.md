# Study on stratified turbulence forced by large scale poloidal modes

Grid N (related to Fh) and Rb (buoyancy Reynolds number).

Standard viscosity (to get a real Kolmogorov scale) and order 4 hyperviscosity:

```
nu_4 ~ injection_rate_4 ** (1 / 3) / k_max ** (10 / 3)
```

## nh = 320 

We start by small simulations for `nh = 320`. The couples `N, Rb` are defined
in `util.py`.

The aspect ratio and t_end are set by something in `get_ratio_nh_nz` and `get_t_end` in  `util.py`):

These simulations are done on the Licallo or Azzurra clusters on 40 cores maximum (1 node).

| N | <15         | 20, 30, 40 | 80, 100, 120 |
|---|-------------|------------|--------------|
|   | 320x320x160 | 320x320x80 | 320x320x40   | 

## nh = 640 

The resolution is doubled in all direction (8 times more grid points) and the
hyperviscosity is decreased by 10 (`~2**(10/3)`).

| N | <15         | 20, 30, 40 | 80, 100, 120 |
|---|-------------|------------|--------------|
|   | 640x640x320 | 640x640x160 | 640x640x80  | 

These simulations are done on the Licallo or Azzurra clusters on 40 cores maximum (1 node).

## nh = 1280 

The resolution is multiplied by 2 and the hyperviscosity is decreased by `2**(10/3)`.

Without projection
| N            | 10            | 20            | 30            | 40             | 80, 100, 120  |
|--------------|---------------|---------------|---------------|----------------|---------------|
|              | 1280x1280x640 | 1280x1280x320 | 1280x1280x320 | 1280x1280x320  | 1280x1280x160 |
| Rb           | 160           | 40, 80, 160   | 20, 40        | 10, 20, 40, 80 | 10            |
| # nodes      | 16            | 8             | 8             | 8              | 4             |
| # cores      | 640           | 320           | 320           | 320            | 160           |
| fft lib      | fftw1d        | fftw1d        | fftw1d        | fftw1d         | fftw1d        |  
| time / simul | 1.25 days     | 2.5 days      | 2.5 days      | 2 days         | 2 days        |
| h.CPU        | 20000         | 20000         | 20000         | 16000          | 8000          |  

With projection
| N            | 10            | 20            | 30            | 40             | 80, 100, 120  |
|--------------|---------------|---------------|---------------|----------------|---------------|
|              | 1280x1280x640 | 1280x1280x320 | 1280x1280x320 | 1280x1280x320  | 1280x1280x160 |
| Rb           | 160           | 40, 80, 160   | 20, 40        | 10, 20, 40, 80 | 10            |
| # nodes      | 16            | 8             | 8             | 8              | 4             |
| # cores      | 640           | 320           | 320           | 320            | 160           |
| fft lib      | fftw1d        | fftw1d        | fftw1d        | fftw1d         | fftw1d        |  
| time / simul | 1.75 days     | 2 days        | 2 days        | 2.25 days      | 2.25 days     |
| h.CPU        | 27000         | 16000         | 16000         | 18000          | 18000         |

These simulations are done on Jean-Zay cluster.
# TODO: Continue here
## nh = 1920 

The resolution is multiplied by 3/2 and the hyperviscosity is decreased by `(3/2)**(10/3)`.



## nh = 2580

The resolution is multiplied by 3/2 and the hyperviscosity is decreased by `(3/2)**(10/3)`.


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
