# PDRanalyser

[![Model grid DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24352982.svg)](https://doi.org/10.5281/zenodo.24352982)

**Constrain the physical conditions of cold neutral gas by comparing observed H₂ (and HD, C I, CO) excitation with grids of Meudon PDR models.**

PDRanalyser is a Python toolkit with an interactive GUI for interpreting molecular absorption systems, such as H₂-bearing damped Lyman-α absorbers toward quasars and Milky Way sightlines. It reads grids of photodissociation-region (PDR) models computed with the [Meudon PDR code](https://ism.obspm.fr/pdr.html), calculates the column densities of individual rotational levels for a given total column, and fits them to the observed values. The fit returns likelihood maps and best-fitting estimates of the **gas number density** and the **intensity of the UV radiation field**, along with other model parameters such as metallicity, cosmic-ray ionization rate and CMB temperature.

## Features

- **Reads Meudon PDR models.** Loads HDF5 outputs of the Meudon PDR code (v1.5.4) and gives access to depth profiles of temperature, pressure, densities, level populations, column densities and heating/cooling rates.
- **Compares excitation diagrams.** Plots the excitation of observed H₂, HD, C I and CO rotational levels against model predictions, with reference lines for thermal (T_exc) and CMB excitation.
- **Fits model grids.** Computes the likelihood across a grid of models (e.g. n_H vs. UV field), with optional systematic uncertainty, and shows the constraint in the 2D parameter plane.
- **Interpolates and regrids.** Refines coarse model grids by interpolation and produces smooth posterior maps and best-fit parameters.
- **Provides an interactive GUI.** A PyQt5/pyqtgraph interface for choosing an absorption system, browsing the model grid, clicking models to inspect their excitation, and exporting results.

## Repository contents

| File | Description |
|---|---|
| `PDR_exc.py` | Core library: the `model` class (reading a single PDR model) and the `PDR_exc` class (grids of models, comparison with observations, likelihood calculation, plotting) |
| `PDRviewer.py`, `__main__.py` | Interactive GUI for exploring model grids and fitting observed systems |
| `PDR_database_public.fits` | Database of published absorption systems (see below) |
| `PDR_database.py` | Reads, views and edits the database with pandas; add systems, components, values and groups |
| `PDR_systems.py` | Classes for observed systems (`qso`, `sy`, `sample`) used by `PDR_exc` |
| `PDR_database_viewer.ipynb` | Notebook to browse the database: groups, tables, single systems, plots, editing |
| `make_public_lists.py` | Lists of published systems; builds `PDR_database_public.fits` from the authors' full database |
| `PDR_tools.py` | Helpers from the `spectro` package: asymmetric uncertainties (`a`), 1D/2D likelihood distributions (`distr1d`, `distr2d`), species container (`e`) |
| `pyratio.py` | Level populations of C I, CO and other species (`spectro.pyratio`, unchanged except for its import block); atomic data in `data/pyratio/` |
| `energy_X_H2.dat` | Energies of rotational/vibrational levels of the ground electronic state of H₂ (E. Roueff) |
| `styles.ini`, `images/` | GUI style sheet and icons |

## Requirements

- Python 3
- `numpy`, `scipy`, `matplotlib`, `pandas`, `astropy`, `h5py`
- `PyQt5`, `pyqtgraph` (for the GUI)
- Optional: `mendeleev` (element data in `e.mendeleev()`), `plotly` (3D plots), `emcee`, `corner`, `chainconsumer` (MCMC in `pyratio.calc_MCMC`)

The `spectro` package is not needed: the parts used are in `PDR_tools.py` and `pyratio.py`.

## Usage

The grid of Meudon PDR models (`.hdf5`, 11.6 GB) is not included in this repository. Download it from Zenodo:

> Klimenko V. V. (2026), *Meudon PDR code (v1.5.4) model grid: gas density, A_V and UV radiation field*, Zenodo, [doi:10.5281/zenodo.24352982](https://doi.org/10.5281/zenodo.24352982)

Download all files of the record into one folder, check them and unpack:

```bash
sha256sum -c SHA256SUMS.txt
for f in grid-n-av-uv_part*.tar; do tar -xf "$f"; done     # creates grid-n-av-uv/ (326 models)
```

Then either move it into this repository as `data/models` (`mv grid-n-av-uv /path/to/PDRanalyser/data/models`), or set the environment variable `PDR_MODELS` to its path:

```bash
export PDR_MODELS=/path/to/grid-n-av-uv
```

**Launch the GUI** (from this folder):

```bash
export PDR_MODELS=/path/to/grid-n-av-uv     # not needed if the models are in data/models
python __main__.py
```

**Use it as a library (e.g. in a Jupyter notebook):**

```python
from PDR_exc import PDR_exc
import matplotlib.pyplot as plt

H2 = PDR_exc(folder='/path/to/grid-n-av-uv/')   # Meudon PDR models from Zenodo; observed systems: H2database='public'
H2.readfolder()

# fit the H2 excitation of an absorption system (component 4 of J0000+0048) over a grid of (UV field, density),
# with the metallicity fixed and 0.1 dex systematic uncertainty
H2.comparegrid('J0000+0048_4', pars=['uv', 'n0'], fixed={'z': 0.160}, syst=0.1)
plt.show()
```

## Database of observed systems

`PDR_database_public.fits` contains the absorption systems from Klimenko et al. (2024, MNRAS 533, 1367):

| group | systems | | `PDR_exc` keyword |
|---|---|---|---|
| `MW_CO_Klimenko2024` | 28 | Milky Way CO sightlines (Table 1) | `'MW_CO'` |
| `MW_CI_Klimenko2024` | 56 | Milky Way C I sightlines from Jenkins & Tripp (2011) (Table A1) | `'MW_CI'` |
| `QSO_Klimenko2024` | 7 | CO-bearing DLAs at high redshift (Table 1) | `'QSO_CO'` |

`PDR_exc` loads all of them by default (`H2database='public'`), or one group with its keyword or group name.

The file has four tables, read as pandas DataFrames: `systems`, `components`, `values` (one row per quantity: column density, T01, n, ...; `type` is `m` measurement, `u` upper limit, `l` lower limit, `f` fixed) and `groups` (which systems and components belong to each group). Values are stored once, so an edit applies to every group that contains the system. You can add your own systems and groups and save them to a new file:

```python
from PDR_database import PDRdatabase

db = PDRdatabase.read('PDR_database_public.fits')
db.groups_list()
db.table('QSO_Klimenko2024', ['H2', 'CO', 'CI'])                       # one row per system
db.table('QSO_Klimenko2024', ['H2', 'T01'], comp=True, errors=True)   # one row per component, with errors

db.add_system('J1234+5678', z_em=2.9, z_dla=2.41, ref=['Me2026'])
c = db.add_component('J1234+5678', z=2.4105)
db.set_value('J1234+5678', 'H2j0', 18.7, 0.05, comp=c)
db.set_value('J1234+5678', 'CO', 13.0, comp=c, type='u')               # upper limit
db.add_group('my_sample', ['J0000+0048', 'J1234+5678'])
db.write('my_database.fits')

H2 = PDR_exc(folder='/path/to/grid-n-av-uv/', H2database='my_sample', database='my_database.fits')
```

See `PDR_database_viewer.ipynb` for more examples.

## Related publications

This code was used to derive physical conditions in H₂-bearing absorbers in, e.g.:

- Klimenko V. V. & Balashev S. A. (2020), *MNRAS*, 498, 1531: *Estimating the UV radiation field and gas density in H₂-bearing DLAs*
- Klimenko V. V., Balashev S. A., Noterdaeme P., Srianand R., Ivanchik A. V. (2024), *MNRAS*, 533, 1367: *Excitation of CO molecules in diffuse gas over cosmic history*
- *(add further papers)*

If you use this code, please cite the relevant paper(s) above. If you use the model grid, please also cite the Zenodo record (doi:10.5281/zenodo.24352982) and the Meudon PDR code (Le Petit et al. 2006, ApJS, 164, 506).

## Authors

Viacheslav Klimenko, with contributions from Sergei Balashev.
