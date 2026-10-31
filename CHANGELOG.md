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
- `size_average.forward_grid` and `size_average.merge_forward`: scattering matrix
  on a fine geometric angular grid up to 2° merged with the regular grid, to
  resolve the forward diffraction peak of large particles (the 0.05° grid gave
  $\frac{1}{2}\int F_{11}\sin\theta\,d\theta \approx 0.53\,C_{sca}$ for a
  median radius of 1 mm). Documented in the theory page.
- Test suite (`tests/`, `pytest`): scattering matrix, efficiencies and
  `S1S2Function` against PyMieScatt, $C_{sca}$ from $S_{11}$, size
  distributions, size averaging against direct integration over the radius, and
  forward-peak normalization.
- `refractive_index.water_refractive_index`: refractive index of water
  (Quan and Fry 1997, Li et al. 2015, Kedenburg et al. 2012, Max and Chapados 2009,
  Hale and Querry 1973, composite of Harmel et al. 2018), used by the beads and
  formazin scripts. The third-party `refractiveindex.info` database is not
  tracked.

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
- `Processor.ScatMat` computes the angular functions $\pi_n, \tau_n$ for all the
  angles at once (recurrence over $n$ vectorized over the angles) instead of one
  call of PyMieScatt `MiePiTau` per angle: ~13× faster, identical results
  (relative difference < 1e-13).
- `generate_osoaa_scatmat.py` merges the forward peak computed on
  `forward_grid` (`data/mueller_mie_microplastic_v4_forward.nc`, generated on the
  first run) into the size-averaged matrices: the OSOAA files are now on a
  non-uniform angular grid and their side and backward scattering is no longer
  overestimated for large particles.
- `Qlut_mie.py` only generates (or reads) the efficiency look-up table on
  `(x, mr, mi)` and plots it; the analysis copied from `plastic_mie.py`, which
  needed an undefined scattering-matrix LUT, was removed.
- Data paths of `junge.py`, `beads.py`, `check_lut.py` and `test_simu.py`
  updated to `data/`.
- Multiprocessing pools are opened in a context manager with `Processor.processes`
  workers, instead of a hard-coded 38 workers.

### Fixed

- `Processor.S1S2Function` returned the undefined variables `SL`, `SR`, `SU`
  (`NameError`) and stored the complex amplitudes `S1`, `S2` in real arrays. It now
  matches PyMieScatt `ScatteringFunction`, including the `normalization` option.
- `Processor.Qeff_compute` no longer writes to `/data/tmp/tmp_mieperso.txt`.
- Plotting loops over `groupby('x')` / `groupby('nr')` (2-D arrays with recent
  xarray) in `formazin_mie.py`, `plastic_mie.py`,
  `formazin_mie_hyperspectral.py` and `backscattering_calib_beads.py`.
- `formazin_mie_hyperspectral.py`: the mean cross section `Csca_hyp` used radii in
  nm with a median radius in µm; the monodisperse case used the array of LUT radii
  instead of the bead radius.
- `formazin_mie_hyperspectral.py`, `backscattering_calib_beads.py`: the
  size-distribution loops used the median radius instead of each radius for
  $\pi r^2$, and the salinity loops did not update the wavelength in water.
- `backscattering_calib_beads.py`: bead radius in µm (`radius_nm * 1e3` →
  `* 1e-3`), the figure name of the angular convolution (`angle` overwritten by a
  list), and the salinity plot no longer uses measurements that are not
  loaded in this script.
- `water_refractive_index.nair` indexed past the end of its coefficient list.
- `SyntaxWarning: invalid escape sequence` in `SizeParam.to_annotation` and
  `SizeParam.to_annotation_r` (Python ≥ 3.12); the returned labels are unchanged.

## [1.0.0]

- pip-installable package (`pyproject.toml`) and Read the Docs configuration.
- `scipy.integrate.trapz` replaced by `trapezoid` for recent NumPy/SciPy.
- Tutorial notebook (`notebook/mie_tuto.ipynb`).
