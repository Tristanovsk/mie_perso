import os

import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.integrate import trapezoid  # import a single function for integration using trapezoidal rule
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 22})
from lmfit import models, report_fit
import PyMieScatt as ps


class Processor():
    r'''Multi-processor Mie computations based on PyMieScatt.

    Each size parameter (or particle diameter) is sent to a different worker.
    The equations are detailed in the *Theoretical background* page of the
    documentation (Mie, 1908; Bohren and Huffman, 1983; Sumlin et al., 2018).

    Parameters
    ----------
    processes : int, optional
        Number of worker processes; defaults to the number of CPUs.
    '''

    def __init__(self, processes=None):
        self.processes = processes or os.cpu_count()

    def S1S2Function(self, m, wavelength, diameter, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5,
                     space='theta',
                     angleMeasure='radians', normalization=None):
        r'''Angular scattering intensities of a single sphere.

        Same as PyMieScatt ``ScatteringFunction``
        (http://pymiescatt.readthedocs.io/en/latest/forward.html#ScatteringFunction):

        .. math:: SL(\theta) = |S_1|^2, \qquad SR(\theta) = |S_2|^2, \qquad SU = \tfrac{1}{2}(SR + SL).

        Parameters
        ----------
        normalization : str, optional
            ``'max'``: divided by the maximum;
            ``'total'``: divided by the integral over the angles.

        Returns
        -------
        tuple
            ``measure`` (angles or q-space), SL, SR, SU.
        '''
        nMedium = nMedium.real
        m /= nMedium
        wavelength /= nMedium
        x = np.pi * diameter / wavelength

        _steps = int(1 + (maxAngle - minAngle) / angularResolution)  # default 361

        if angleMeasure in ['radians', 'RADIANS', 'rad', 'RAD']:
            adjust = np.pi / 180
        elif angleMeasure in ['gradians', 'GRADIANS', 'grad', 'GRAD']:
            adjust = 1 / 200
        else:
            adjust = 1

        if space in ['q', 'qspace', 'QSPACE', 'qSpace']:
            # _steps *= 10
            _steps += 1
            if minAngle == 0:
                minAngle = 1e-5
            # measure = np.logspace(np.log10(minAngle),np.log10(maxAngle),_steps)*np.pi/180
            measure = np.linspace(minAngle, maxAngle, _steps) * np.pi / 180
            _q = True
        else:
            measure = np.linspace(minAngle, maxAngle, _steps) * adjust
            _q = False
        if x == 0:
            return measure, 0, 0, 0
        _measure = np.linspace(minAngle, maxAngle, _steps) * np.pi / 180
        S1, S2 = np.zeros(_steps, dtype=complex), np.zeros(_steps, dtype=complex)

        for j in range(_steps):
            u = np.cos(_measure[j])
            S1[j], S2[j] = ps.MieS1S2(m, x, u)

        SL = np.abs(S1) ** 2
        SR = np.abs(S2) ** 2
        SU = (SR + SL) / 2

        if normalization in ['m', 'M', 'max', 'MAX']:
            SL /= np.max(SL)
            SR /= np.max(SR)
            SU /= np.max(SU)
        elif normalization in ['t', 'T', 'total', 'TOTAL']:
            SL /= trapezoid(SL, measure)
            SR /= trapezoid(SR, measure)
            SU /= trapezoid(SU, measure)

        if _q:
            measure = (4 * np.pi / wavelength) * np.sin(measure / 2) * (diameter / 2)
        return measure, SL, SR, SU

    def coerceDType(self, d):
        if type(d) is not np.ndarray:
            return np.array(d)
        else:
            return d

    def define_scat_ang(self, steps=360, measure='radians'):
        _steps = int((steps) / 2) + 1
        adjust = 1.
        if measure != 'radians':
            adjust = 180. / np.pi
        return np.concatenate(
            [np.arcsin(np.linspace(0, 1, _steps)), -np.arcsin(np.linspace(1, 0, _steps)[1:]) + np.pi]) * adjust

    def func(self, p):
        m, wavelength, d, n, minAngle, maxAngle, angularResolution, space = p
        kwargs = {'minAngle': minAngle,
                  'maxAngle': maxAngle,
                  'angularResolution': angularResolution,
                  'space': space,
                  'normalization': None}
        measure, l, r, u = ps.ScatteringFunction(m, wavelength, d, **kwargs)
        SL = l * n
        SR = r * n
        SU = u * n
        return SL, SR, SU

    def SF_SD_mp(self, m, wavelength, dp, ndp, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5,
                 space='theta', angleMeasure='radians', normalization=None):
        r'''Angular scattering intensities of a size distribution, in parallel.

        Multi-processor version of PyMieScatt ``SF_SD``: the intensities of each
        diameter :math:`d_p`, weighted by the number of particles :math:`n(d_p)`,
        are summed over the distribution,

        .. math:: SL(\theta) = \sum_{d_p} |S_1(\theta; d_p)|^2\, n(d_p), \quad
                  SR(\theta) = \sum_{d_p} |S_2(\theta; d_p)|^2\, n(d_p), \quad
                  SU = \tfrac{1}{2}(SR + SL).

        Parameters
        ----------
        m : complex
            Refractive index of the particles (relative to vacuum if ``nMedium``
            is given, it is then divided by ``nMedium``).
        wavelength : float
            Wavelength in vacuum (same unit as ``dp``).
        dp, ndp : array_like
            Particle diameters and number of particles per diameter bin.
        normalization : str, optional
            ``'number'``: divided by :math:`\int n(d_p)\,dd_p`;
            ``'max'``: divided by the maximum;
            ``'total'``: divided by the integral over the angles.

        Returns
        -------
        tuple
            ``measure`` (angles or q-space), SL, SR, SU.
        '''
        #  http://pymiescatt.readthedocs.io/en/latest/forward.html#SF_SD
        nMedium = nMedium.real
        m /= nMedium
        wavelength /= nMedium

        _steps = int(1 + (maxAngle - minAngle) / angularResolution)
        ndp = self.coerceDType(ndp)
        dp = self.coerceDType(dp)

        kwargs = {'minAngle': minAngle,
                  'maxAngle': maxAngle,
                  'angularResolution': angularResolution,
                  'space': space,
                  'normalization': None}

        measure, l, r, u = ps.ScatteringFunction(m, wavelength, dp[0], **kwargs)
        p = []
        for n, d in zip(ndp, dp):
            p.append([m, wavelength, d, n, minAngle, maxAngle, angularResolution, space])

        with multiprocessing.Pool(processes=self.processes) as pool:
            res = np.array(pool.map(self.func, p))

        SL = np.sum(res[:, 0], axis=0)
        SR = np.sum(res[:, 1], axis=0)
        SU = np.sum(res[:, 2], axis=0)

        if normalization in ['n', 'N', 'number', 'particles']:
            _n = trapezoid(ndp, dp)
            SL /= _n
            SR /= _n
            SU /= _n
        elif normalization in ['m', 'M', 'max', 'MAX']:
            SL /= np.max(SL)
            SR /= np.max(SR)
            SU /= np.max(SU)
        elif normalization in ['t', 'T', 'total', 'TOTAL']:
            SL /= trapezoid(SL, measure)
            SR /= trapezoid(SR, measure)
            SU /= trapezoid(SU, measure)
        return measure, SL, SR, SU

    def ScatMat(self, p):
        r'''Non-null scattering matrix terms of a single sphere.

        The amplitude functions are computed from the Mie coefficients
        :math:`a_n, b_n` and the angular functions :math:`\pi_n, \tau_n`,

        .. math:: S_1 = \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)} (a_n \pi_n + b_n \tau_n), \qquad
                  S_2 = \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)} (a_n \tau_n + b_n \pi_n),

        with :math:`n_{max} = x + 4x^{1/3} + 2`, and the scattering matrix
        elements are (Bohren and Huffman, 1983, Eq. 4.77)

        .. math:: S_{11} = \tfrac{1}{2}(|S_2|^2 + |S_1|^2), \quad
                  S_{12} = \tfrac{1}{2}(|S_2|^2 - |S_1|^2), \quad
                  S_{33} = \tfrac{1}{2}(S_2^* S_1 + S_2 S_1^*), \quad
                  S_{34} = \tfrac{i}{2}(S_1 S_2^* - S_2 S_1^*).

        They are dimensionless: the differential scattering cross section is
        :math:`S_{11}/k^2` and :math:`C_{sca} = \frac{\lambda^2}{2\pi}\int_0^\pi S_{11}\sin\theta\,d\theta`.
        For :math:`x < 0.5`, PyMieScatt ``AutoMie_ab`` uses the low-frequency
        expansion (:math:`a_1, a_2, b_1` only).

        Parameters
        ----------
        p : sequence
            ``(m, x, u)`` with ``m`` the complex refractive index relative to
            the medium (``n + ik``, ``k > 0`` for absorption), ``x`` the size
            parameter (``2 pi r / lambda``) and ``u`` the array of cosines of
            the scattering angles.

        Returns
        -------
        tuple of numpy.ndarray
            S11, S12, S33, S34 (dimensionless, from the amplitudes S1, S2).
        '''
        m, x, u = p
        u = np.asarray(u, dtype=float)
        S1, S2 = np.zeros(u.shape, dtype=complex), np.zeros(u.shape, dtype=complex)

        an, bn = ps.AutoMie_ab(m, x)
        # upward recurrence of pi_n, tau_n (as in PyMieScatt MiePiTau),
        # vectorized over the angles and summed on the fly
        pi_prev, pi_n = np.zeros_like(u), np.ones_like(u)
        for n in range(1, len(an) + 1):
            if n > 1:
                pi_prev, pi_n = pi_n, ((2 * n - 1) * u * pi_n - n * pi_prev) / (n - 1)
            tau_n = n * u * pi_n - (n + 1) * pi_prev
            n2 = (2 * n + 1) / (n * (n + 1))
            S1 += n2 * (an[n - 1] * pi_n + bn[n - 1] * tau_n)
            S2 += n2 * (an[n - 1] * tau_n + bn[n - 1] * pi_n)
        S11 = (0.5 * (np.abs(S2) ** 2 + np.abs(S1) ** 2)).real
        S12 = (0.5 * (np.abs(S2) ** 2 - np.abs(S1) ** 2)).real
        S33 = (0.5 * (np.conjugate(S2) * S1 + S2 * np.conjugate(S1))).real
        S34 = (0.5j * (S1 * np.conjugate(S2) - S2 * np.conjugate(S1))).real
        return S11, S12, S33, S34

    def ScatMat_mp(self, m, x, theta):
        r'''Scattering matrix for a series of size parameters, in parallel.

        See :meth:`ScatMat` for the equations. The matrix of a population of
        particles of number size distribution :math:`n(r)` is obtained by
        integrating the output over the radius :math:`r = x\lambda/(2\pi)`:

        .. math:: F_{ij}(\theta) = \frac{\lambda^2}{\pi}
                  \frac{\int S_{ij}(x(r), \theta)\, n(r)\, dr}{\int n(r)\, dr},
                  \qquad \frac{1}{2}\int_0^\pi F_{11}\sin\theta\,d\theta = \langle C_{sca} \rangle.

        Parameters
        ----------
        m : complex
            Refractive index of the particles relative to the medium
            (``n + ik``, ``k > 0`` for absorption).
        x : array_like
            Size parameters ``pi * diameter / wavelength`` (unitless), with the
            wavelength taken in the medium.
        theta : array_like
            Scattering angles in radians.

        Returns
        -------
        xarray.Dataset
            S11, S12, S33, S34 on the ``(x, theta)`` dimensions.
        '''

        u = np.cos(theta)
        p = []
        for ix, x_ in enumerate(x):
            p.append([m, x_, u])

        with multiprocessing.Pool(processes=self.processes) as pool:
            res = np.array(pool.map(self.ScatMat, p))
        S11, S12, S33, S34 = res[:, 0], res[:, 1], res[:, 2], res[:, 3]

        mueller = xr.Dataset(data_vars=dict(S11=(['x', 'theta'], S11),
                                            S12=(['x', 'theta'], S12),
                                            S33=(['x', 'theta'], S33),
                                            S34=(['x', 'theta'], S34)),
                             coords=dict(x=x, theta=theta, nr=m.real, ni=m.imag),
                             attrs=dict(
                                 description='non-null scattering matrix terms from Mie computations using PyMieScatt'))
        return mueller

    def Qeff_compute(self, p):
        r'''Efficiency factors of a single sphere (PyMieScatt ``MieQ``).

        .. math:: Q_{ext} = \frac{2}{x^2}\sum_{n=1}^{n_{max}} (2n+1)\,\mathrm{Re}\{a_n + b_n\}, \qquad
                  Q_{sca} = \frac{2}{x^2}\sum_{n=1}^{n_{max}} (2n+1)(|a_n|^2 + |b_n|^2),

        .. math:: Q_{abs} = Q_{ext} - Q_{sca}, \qquad
                  Q_{back} = \frac{1}{x^2}\left|\sum_{n=1}^{n_{max}} (2n+1)(-1)^n (a_n - b_n)\right|^2, \qquad
                  Q_{pr} = Q_{ext} - g\,Q_{sca},

        and :math:`Q_{ratio} = Q_{back}/Q_{sca}`. The Rayleigh approximation is
        used for :math:`x \le 0.05`. Cross sections are :math:`C = Q\,\pi r^2`.

        Parameters
        ----------
        p : sequence
            ``(x, mr, mi)``: size parameter, real and imaginary parts of the
            relative refractive index.

        Returns
        -------
        list
            ``[x, mr, mi, Qext, Qsca, Qabs, g, Qpr, Qback, Qratio]``
        '''
        # fake wavelength to compute diameter
        wl = 500
        x, mr, mi = p

        m = complex(mr, mi)
        diameter = wl * x / np.pi
        return [x, mr, mi, *ps.MieQ(m, wl, diameter)]

    def Qeff_mp(self, m, x):
        r'''Efficiency factors for a series of size parameters, in parallel.

        See :meth:`Qeff_compute` for the equations. The mean cross sections of
        a population of particles are
        :math:`\langle C \rangle = \int Q(x(r))\,\pi r^2 n(r)\,dr / \int n(r)\,dr`.

        Parameters
        ----------
        m : complex
            Refractive index of the particles relative to the medium.
        x : array_like
            Size parameters (unitless).

        Returns
        -------
        xarray.Dataset
            Qext, Qsca, Qabs, g, Qpr, Qback, Qratio on the ``x`` dimension.
        '''
        p = [[x_, m.real, m.imag] for x_ in x]
        with multiprocessing.Pool(processes=self.processes) as pool:
            res = pool.map(self.Qeff_compute, p)
        df = pd.DataFrame(res, columns=['x', 'nr', 'ni', 'Qext', 'Qsca', 'Qabs', 'g', 'Qpr', 'Qback', 'Qratio'])
        return (df.drop(columns=['nr', 'ni']).set_index('x').to_xarray()
                .assign_coords(nr=m.real, ni=m.imag))
