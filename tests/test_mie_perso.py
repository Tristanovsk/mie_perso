'''Tests of mie_perso against PyMieScatt and direct numerical integration.

Run with ``pytest`` from the repository root.
'''

import os

import numpy as np
import pytest
import xarray as xr
import PyMieScatt as ps
from scipy.integrate import trapezoid

from mie_perso import mie_multiprocess, psd as psd_module, size_average

PSD = psd_module.PSD()


@pytest.fixture(scope='module')
def pmie():
    return mie_multiprocess.Processor(processes=min(8, os.cpu_count()))


# -------------------------------------
# single particle
# -------------------------------------
@pytest.mark.parametrize('m', [1.2 + 0j, 1.05 + 0.01j])
@pytest.mark.parametrize('x', [0.3, 5., 80.])
def test_scatmat_vs_pymiescatt(pmie, m, x):
    wl = 0.5
    diameter = x * wl / np.pi
    mu = np.cos(np.linspace(0, np.pi, 91))
    S11, S12, S33, S34 = pmie.ScatMat((m, x, mu))
    ref = np.array([ps.MatrixElements(m, wl, diameter, mu_) for mu_ in mu]).T
    for i, S in enumerate([S11, S12, S33, S34]):
        np.testing.assert_allclose(S, ref[i], rtol=1e-9, atol=1e-12 * S11.max())


@pytest.mark.parametrize('m, x', [(1.2 + 0j, 3.), (1.05 + 0.01j, 40.)])
def test_csca_from_s11(pmie, m, x):
    '''C_sca = lambda^2/(2 pi) int S11 sin(theta) dtheta (B&H Eq. 3.12 and 4.77).'''
    wl = 0.5
    theta = np.linspace(0, np.pi, 20001)
    S11 = pmie.ScatMat((m, x, np.cos(theta)))[0]
    Csca = wl ** 2 / (2 * np.pi) * trapezoid(S11 * np.sin(theta), theta)
    Qsca = ps.MieQ(m, wl, x * wl / np.pi)[1]
    np.testing.assert_allclose(Csca, Qsca * np.pi * (x * wl / (2 * np.pi)) ** 2, rtol=1e-5)


def test_qeff_compute(pmie):
    x, mr, mi = 12., 1.15, 0.001
    res = pmie.Qeff_compute([x, mr, mi])
    ref = ps.MieQ(complex(mr, mi), 500, 500 * x / np.pi)
    np.testing.assert_allclose(res[3:], ref)


@pytest.mark.parametrize('normalization', [None, 'max', 'total'])
@pytest.mark.parametrize('space', ['theta', 'qspace'])
def test_s1s2function_vs_scatteringfunction(pmie, normalization, space):
    args = (1.5 + 0.01j, 532, 1000)
    kw = dict(nMedium=1.33, normalization=normalization, space=space)
    res = pmie.S1S2Function(*args, **kw)
    ref = ps.ScatteringFunction(*args, **kw)
    for a, b in zip(res, ref):
        np.testing.assert_allclose(a, b, rtol=1e-10)


def test_scatmat_mp(pmie):
    x = np.array([0.5, 2., 10.])
    theta = np.linspace(0, np.pi, 19)
    res = pmie.ScatMat_mp(1.1 + 0j, x, theta)
    assert res.S11.dims == ('x', 'theta')
    np.testing.assert_allclose(res.S11.sel(x=2.), pmie.ScatMat((1.1 + 0j, 2., np.cos(theta)))[0])


# -------------------------------------
# size distributions
# -------------------------------------
def test_lognorm_normalized():
    r = np.logspace(-4, 4, 20001)
    assert trapezoid(PSD.lognorm(r, rn_med=1.5, sigma=0.6), r) == pytest.approx(1, rel=1e-6)


def test_sizeparam_annotations_compile():
    '''No invalid escape sequences (SyntaxWarning, error in future Python versions).'''
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with open(psd_module.__file__) as f:
            compile(f.read(), psd_module.__file__, 'exec')
    r = np.logspace(-2, 2, 2001)
    sp = psd_module.SizeParam(r, PSD.lognorm(r, rn_med=1, sigma=0.5))
    assert isinstance(sp.to_annotation(), str) and isinstance(sp.to_annotation_r(), str)


# -------------------------------------
# size averaging
# -------------------------------------
def test_trapezoid_weights():
    x = np.sort(np.random.default_rng(0).uniform(0, 10, 50))
    f = np.sin(x)
    assert np.sum(size_average.trapezoid_weights(x) * f) == pytest.approx(trapezoid(f, x))


@pytest.fixture(scope='module')
def lut(pmie):
    x = np.linspace(0.1, 30, 300)
    theta = np.linspace(0, np.pi, 361)
    mueller = xr.concat([pmie.ScatMat_mp(complex(nr, 0), x, theta) for nr in [1.05, 1.2]], 'nr')
    Qeff = xr.concat([pmie.Qeff_mp(complex(nr, 0), x) for nr in [1.05, 1.2]], 'nr')
    return mueller, Qeff


def _direct(lut, wavelength, rn_med, sigma, n_medium):
    '''Reference: integration over the radius, as in the scripts before size_average.'''
    mueller, Qeff = lut
    wl = wavelength / n_medium
    radius = mueller.x * wl / (2 * np.pi)
    ndp = PSD.lognorm(radius, rn_med=rn_med, sigma=sigma)
    norm = trapezoid(ndp, radius)
    S = (mueller * ndp).assign_coords(radius=radius).swap_dims(x='radius').integrate('radius') / norm
    Csca = (Qeff.Qsca * np.pi * radius ** 2 * ndp).assign_coords(radius=radius).swap_dims(
        x='radius').integrate('radius') / norm
    return wl ** 2 / np.pi * S, Csca


def test_integrate_scatmat_and_cross_section(lut):
    mueller, Qeff = lut
    wavelengths, rmeds, sigma, n_medium = np.array([0.4, 0.6]), [0.3, 1.], 0.4, 1.334
    w = size_average.psd_family_weights(mueller.x, wavelengths,
                                        lambda r, rmed: PSD.lognorm(r, rn_med=rmed, sigma=sigma),
                                        rmeds, 'r_med', n_medium=n_medium)
    np.testing.assert_allclose(w.sum('x'), 1)
    F = size_average.integrate_scatmat(mueller.chunk({'nr': 1}), w).compute()
    C = size_average.integrate_cross_section(Qeff, w)
    for wl in wavelengths:
        for rmed in rmeds:
            F_ref, Csca_ref = _direct(lut, wl, rmed, sigma, n_medium)
            xr.testing.assert_allclose(F.sel(wavelength=wl, r_med=rmed, drop=True).drop_vars('ni'),
                                       F_ref.drop_vars('ni'))
            np.testing.assert_allclose(C.Csca.sel(wavelength=wl, r_med=rmed), Csca_ref, rtol=1e-10)
            # phase function normalization: 1/2 int F11 sin(theta) dtheta = <Csca> (0.5 deg grid)
            F11 = F.S11.sel(wavelength=wl, r_med=rmed)
            np.testing.assert_allclose((F11 * np.sin(F11.theta)).integrate('theta') / 2, Csca_ref, rtol=5e-3)


def test_mean_asymmetry(lut):
    mueller, Qeff = lut
    w = size_average.psd_weights(mueller.x, 0.5, lambda r: PSD.lognorm(r, rn_med=0.5, sigma=0.3))
    F11 = size_average.integrate_scatmat(mueller, w).S11.isel(wavelength=0)
    theta = F11.theta
    g = (F11 * np.cos(theta) * np.sin(theta)).integrate('theta') / (F11 * np.sin(theta)).integrate('theta')
    np.testing.assert_allclose(size_average.mean_asymmetry(Qeff, w).isel(wavelength=0), g, rtol=2e-3)


def test_merge_forward(pmie):
    '''A fine forward grid restores the normalization for large x.

    Narrow size distribution around x = 3000 (forward peak of ~0.02 deg): the
    diffraction rings of a single size are not resolved by the 0.05 deg grid.
    '''
    x = np.linspace(2700, 3300, 31)
    m = 1.15 + 0j
    theta = np.linspace(0, np.pi, 3601)
    theta_fwd = size_average.forward_grid()
    S = pmie.ScatMat_mp(m, x, theta)
    S_fwd = pmie.ScatMat_mp(m, x, theta_fwd)
    merged = size_average.merge_forward(S, S_fwd)
    assert merged.theta.size == (theta > theta_fwd[-1]).sum() + theta_fwd.size
    assert np.all(np.diff(merged.theta) > 0)
    # 1/2 int S11 sin dtheta = x^2 Qsca / 4 for each size
    Qsca = np.array([ps.MieQ(m, 1., x_ / np.pi)[1] for x_ in x])
    ref = np.mean(x ** 2 * Qsca / 4)
    half_int = lambda S11: float((S11 * np.sin(S11.theta)).integrate('theta').mean('x') / 2)
    assert abs(half_int(merged.S11) / ref - 1) < 2e-3
    assert half_int(S.S11) / ref < 0.9
