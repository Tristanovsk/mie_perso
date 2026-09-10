import os

import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.integrate import trapz  # import a single function for integration using trapezoidal rule
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


class processor():
    def __init__(self):
        pass

    def S1S2Function(self, m, wavelength, diameter, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5,
                     space='theta',
                     angleMeasure='radians', normalization=None):
        # TODO check this part to return  SL, SR, SU
        #  http://pymiescatt.readthedocs.io/en/latest/forward.html#ScatteringFunction
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
        S1, S2 = np.zeros(_steps), np.zeros(_steps)

        for j in range(_steps):
            u = np.cos(_measure[j])
            S1[j], S2[j] = ps.MieS1S2(m, x, u)

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

        pool = multiprocessing.Pool(processes=38)
        res = np.array(pool.map(self.func, p))

        SL = np.sum(res[:, 0], axis=0)
        SR = np.sum(res[:, 1], axis=0)
        SU = np.sum(res[:, 2], axis=0)

        if normalization in ['n', 'N', 'number', 'particles']:
            _n = trapz(ndp, dp)
            SL /= _n
            SR /= _n
            SU /= _n
        elif normalization in ['m', 'M', 'max', 'MAX']:
            SL /= np.max(SL)
            SR /= np.max(SR)
            SU /= np.max(SU)
        elif normalization in ['t', 'T', 'total', 'TOTAL']:
            SL /= trapz(SL, measure)
            SR /= trapz(SR, measure)
            SU /= trapz(SU, measure)
        return measure, SL, SR, SU

    def ScatMat(self, p):
        '''

        :param p: array of
                            m: refractive index in medium
                            x: size parameter (2pi r / lambda)
                            u: array of cosine of scattering angles
        :return:
        '''
        m, x, u = p
        Ntheta = len(u)
        S11, S12, S33, S34 = np.zeros([Ntheta]), np.zeros([Ntheta]), np.zeros([Ntheta]), np.zeros([Ntheta])
        for iu, u_ in enumerate(u):
            S1, S2 = ps.MieS1S2(m, x, u_)
            S11[iu] = (0.5 * (np.abs(S2) ** 2 + np.abs(S1) ** 2)).real
            S12[iu] = (0.5 * (np.abs(S2) ** 2 - np.abs(S1) ** 2)).real
            S33[iu] = (0.5 * (np.conjugate(S2) * S1 + S2 * np.conjugate(S1))).real
            S34[iu] = (0.5j * (S1 * np.conjugate(S2) - S2 * np.conjugate(S1))).real
        return S11, S12, S33, S34

    def ScatMat_mp(self, m, x, theta):
        '''

        :param m: complex refractive index of particles
        :param x: size parameter (np.pi * diameter / wavelength unitless)
        :param theta:  scattering angles in radians
        :return: xarray.Dataset of mueller matrix non null elements
        '''

        u = np.cos(theta)
        p = []
        for ix, x_ in enumerate(x):
            p.append([m, x_, u])

        pool = multiprocessing.Pool(processes=38)
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
