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

### Changed

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
