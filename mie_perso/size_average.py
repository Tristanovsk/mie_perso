'''Integration of Mie look-up tables over particle size distributions.

The look-up tables computed with :meth:`mie_perso.Processor.ScatMat_mp` and
:meth:`mie_perso.Processor.Qeff_mp` are indexed by the size parameter
:math:`x = 2\\pi r/\\lambda`. Since the radius is proportional to :math:`x` at a
given wavelength, the size averages

.. math:: \\langle f \\rangle = \\frac{\\int f(x(r))\\, n(r)\\, dr}{\\int n(r)\\, dr}
          = \\sum_x f(x)\\, w_\\lambda(x),
          \\qquad
          w_\\lambda(x) = \\frac{t(x)\\, n(x\\lambda/2\\pi)}{\\sum_x t(x)\\, n(x\\lambda/2\\pi)},

with :math:`t(x)` the trapezoidal weights of the :math:`x` grid, reduce to a
contraction of the table with the weights :math:`w_\\lambda(x)`. All the
wavelengths and size distributions are then computed in a single pass over the
(possibly larger than memory) table, which can be opened lazily with dask, e.g.
``xr.open_dataset(file, chunks={'nr': 4})``.
'''

import numpy as np
import pandas as pd
import xarray as xr


def trapezoid_weights(x):
    '''Weights :math:`t_i` such that ``scipy.integrate.trapezoid(f, x) == sum(t * f)``.

    Parameters
    ----------
    x : array_like
        Monotonic 1-D grid.

    Returns
    -------
    numpy.ndarray
    '''
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


def psd_weights(x, wavelength, psd, n_medium=1.0):
    r'''Normalized size-distribution weights :math:`w_\lambda(x)`.

    Parameters
    ----------
    x : array_like
        Size parameters of the look-up table.
    wavelength : float or array_like
        Wavelength(s) in vacuum, in the unit of the radius expected by ``psd``
        (e.g. µm).
    psd : callable
        Number size distribution :math:`n(r)`, called for each wavelength with
        the 1-D array of radii ``r = x * wavelength / (2 pi n_medium)``.
    n_medium : float, optional
        Refractive index of the medium.

    Returns
    -------
    xarray.DataArray
        Weights on the ``(wavelength, x)`` dimensions, summing to 1 over ``x``,
        with the wavelength in the medium as ``wavelength_medium`` coordinate
        (used by the integration functions). The ``wavelength`` coordinate can be
        relabelled freely, e.g. ``.assign_coords(wavelength=wavelengths_nm)``.
        See :func:`psd_family_weights` to process several distributions in the
        same pass.
    '''
    x = np.asarray(x, dtype=float)
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
    wavelength_medium = wavelength / n_medium
    # psd is called for each wavelength with a 1-D array of radii
    w = np.array([np.asarray(psd(wl * x / (2 * np.pi)), dtype=float) for wl in wavelength_medium])
    w *= trapezoid_weights(x)
    w = np.nan_to_num(w)
    w /= w.sum(axis=1, keepdims=True)
    return xr.DataArray(w, dims=('wavelength', 'x'),
                        coords={'wavelength': wavelength, 'x': x,
                                'wavelength_medium': ('wavelength', wavelength_medium)})


def psd_family_weights(x, wavelength, psd, params, dim, n_medium=1.0):
    r'''Weights of a family of size distributions along a new dimension.

    Parameters
    ----------
    x, wavelength, n_medium :
        See :func:`psd_weights`.
    psd : callable
        ``psd(r, p)`` returning :math:`n(r)` for the parameter value ``p``.
    params : sequence
        Parameter values (e.g. median radii).
    dim : str
        Name of the new dimension (e.g. ``'r_med'``).

    Returns
    -------
    xarray.DataArray
        Weights on the ``(dim, wavelength, x)`` dimensions.

    Examples
    --------
    >>> w = psd_family_weights(mueller.x, wavelengths_nm * 1e-3,
    ...                        lambda r, rmed: PSD().lognorm(r, rn_med=rmed, sigma=0.5),
    ...                        [1., 10.], 'r_med', n_medium=1.334)
    >>> F = integrate_scatmat(mueller, w).compute()
    '''
    return xr.concat([psd_weights(x, wavelength, lambda r, p=p: psd(r, p), n_medium=n_medium)
                      for p in params], dim=pd.Index(params, name=dim))


def _wavelength_medium(weights):
    return weights.wavelength_medium if 'wavelength_medium' in weights.coords else weights.wavelength


def size_average(lut, weights, dim='x'):
    '''Weighted sum of each variable of ``lut`` over ``dim`` (lazy if ``lut`` is dask-backed).

    Uses :func:`xarray.dot` (einsum), which avoids building the intermediate
    product ``lut * weights`` with all the dimensions of both operands.
    '''
    if isinstance(lut, xr.Dataset):
        return lut.map(lambda da: xr.dot(da, weights, dim=dim) if dim in da.dims else da)
    return xr.dot(lut, weights, dim=dim)


def integrate_scatmat(mueller, weights):
    r'''Scattering matrix of a particle population (area unit of :math:`\lambda^2`).

    .. math:: F_{ij}(\theta) = \frac{\lambda^2}{\pi} \sum_x S_{ij}(x, \theta)\, w_\lambda(x),
              \qquad \frac{1}{2}\int_0^\pi F_{11}\sin\theta\, d\theta = \langle C_{sca} \rangle

    with :math:`\lambda` the wavelength in the medium (unit of the weights).

    Parameters
    ----------
    mueller : xarray.Dataset
        Output of :meth:`mie_perso.Processor.ScatMat_mp` (``S11``, ``S12``,
        ``S33``, ``S34`` on ``x``, ``theta`` and possibly ``nr``...).
    weights : xarray.DataArray
        Output of :func:`psd_weights`.
    '''
    return _wavelength_medium(weights) ** 2 / np.pi * size_average(mueller, weights)


def integrate_cross_section(Qeff, weights, variables=('Qext', 'Qsca', 'Qabs', 'Qback')):
    r'''Mean cross sections of a particle population (unit of :math:`\lambda^2`).

    .. math:: \langle C \rangle = \sum_x Q(x)\, \pi r(x)^2\, w_\lambda(x),
              \qquad r(x) = \frac{x \lambda}{2\pi}

    Parameters
    ----------
    Qeff : xarray.Dataset
        Output of :meth:`mie_perso.Processor.Qeff_mp`.
    weights : xarray.DataArray
        Output of :func:`psd_weights`.
    variables : sequence of str
        Efficiencies to integrate; ``Q...`` is renamed ``C...``.

    Returns
    -------
    xarray.Dataset
    '''
    radius = weights.x * _wavelength_medium(weights) / (2 * np.pi)
    cross = size_average(Qeff[list(variables)], weights * np.pi * radius ** 2)
    return cross.rename({v: 'C' + v[1:] for v in variables})


def mean_asymmetry(Qeff, weights):
    r'''Scattering-weighted asymmetry parameter of a particle population.

    .. math:: \langle g \rangle = \frac{\sum_x g\, Q_{sca}\, r^2\, w_\lambda}{\sum_x Q_{sca}\, r^2\, w_\lambda}
    '''
    w = weights * (weights.x * _wavelength_medium(weights)) ** 2
    return size_average(Qeff.g * Qeff.Qsca, w) / size_average(Qeff.Qsca, w)
