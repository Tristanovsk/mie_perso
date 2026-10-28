# Changelog

All notable changes to `mie_perso` are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Sphinx documentation (`docs/`): README landing page, tutorial notebook rendered
  with MyST-NB, and API reference generated with autosummary.
- *Theoretical background* page (`docs/source/theory.md`) with the numbered
  equations of Mie theory, based on Mie (1908), Mishchenko and Travis (2008) and
  Sumlin et al. (2018): size parameter and relative refractive index, Mie
  coefficients $a_n, b_n$, angular functions $\pi_n, \tau_n$, amplitude functions
  $S_1, S_2$, scattering matrix elements, efficiency factors and asymmetry
  parameter, Rayleigh limit, phase function normalization, integration over a size
  distribution, size distributions and effective radius/variance.
- "Theory in a nutshell" section in the README.
- Equations and NumPy-style docstrings for `Processor`, `PSD` and `SizeParam`.
- `Processor(processes=None)` to set the number of workers (all CPUs by default).
- `Processor.Qeff_mp`: efficiency factors for a series of size parameters, in
  parallel, returned as an `xarray.Dataset`.
- `mie_perso.size_average`: integration of the scattering-matrix and efficiency
  look-up tables over size distributions as a single tensor contraction
  (`xarray.dot`) over the size parameter, for all wavelengths and distributions in
  one (dask-friendly) pass over the table: `psd_weights`, `psd_family_weights`,
  `integrate_scatmat`, `integrate_cross_section`, `mean_asymmetry`.

### Changed

- `generate_osoaa_scatmat.py` uses `mie_perso.size_average`: the LUT is opened
  lazily (`chunks={'nr': 1}`) and integrated once for all wavelengths and median
  radii instead of once per (wavelength, median radius) with
  `integrate('radius').compute()` (76 combinations on the 2.5 GB microplastic LUT:
  ~300 s → ~5 s on 40 CPUs, identical results).
- Same for the analysis scripts `formazin_mie.py`, `plastic_mie.py`, `Qlut_mie.py`,
  `formazin_mie_hyperspectral.py` and `backscattering_calib_beads.py`:
  - the LUTs are opened lazily (`chunks={'nr': 1}`), so that `dropna('x')` no
    longer loads the whole table (2.5–6.3 GB) into memory;
  - the size-distribution integrals are computed once for all the wavelengths and
    distribution parameters before the plotting loops;
  - the angular convolution of `backscattering_calib_beads.py` is a single
    weighted contraction over `theta` instead of a `groupby('theta')` loop;
  - the Qeff look-up tables use `Processor.processes` workers in a closed pool;
  - data paths updated from `mie_perso/data/` to `data/`.
  The scattering matrices of these scripts are now normalized by
  $\int n(r)\,dr$ and expressed in µm² (also for `size_class = 'large'`, where the
  cross sections were previously computed in mm²).
- Multiprocessing pools are opened in a context manager with `Processor.processes`
  workers, instead of a hard-coded 38 workers.

### Fixed

- `Processor.S1S2Function` returned the undefined variables `SL`, `SR`, `SU`
  (`NameError`) and stored the complex amplitudes `S1`, `S2` in real arrays. It now
  matches PyMieScatt `ScatteringFunction`, including the `normalization` option.
- `Processor.Qeff_compute` no longer writes to `/data/tmp/tmp_mieperso.txt`.
- `SyntaxWarning: invalid escape sequence` in `SizeParam.to_annotation` and
  `SizeParam.to_annotation_r` (Python ≥ 3.12); the returned labels are unchanged.

## [1.0.0]

- pip-installable package (`pyproject.toml`) and Read the Docs configuration.
- `scipy.integrate.trapz` replaced by `trapezoid` for recent NumPy/SciPy.
- Tutorial notebook (`notebook/mie_tuto.ipynb`).
