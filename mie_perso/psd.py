import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import trapezoid


class PSD:
    r'''Particle size distributions :math:`n(r)` as functions of the radius :math:`r`.

    See the *Theoretical background* page of the documentation for the
    definitions and the conversions between the log-normal parameters.
    '''

    def __init__(self):
        pass

    def norm(self, r, mu=1, sigma=0.5):
        r'''Normal distribution of mean ``mu`` and standard deviation ``sigma``.

        .. math:: n(r) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left[-\frac{(r-\mu)^2}{2\sigma^2}\right]
        '''
        return (np.exp(-(r - mu) ** 2 / (2 * sigma ** 2))
                / (sigma * np.sqrt(2 * np.pi)))

    def lognorm(self, r, rn_med=1, sigma=0.5):
        r'''Log-normal number distribution normalized to :math:`\int n(r)\,dr = 1`.

        .. math:: n(r) = \frac{1}{r\,\sigma\sqrt{2\pi}}
                  \exp\left[-\frac{(\ln r - \ln r_{n,med})^2}{2\sigma^2}\right]

        Parameters
        ----------
        r : array_like
            Radius.
        rn_med : float
            Number median radius :math:`r_{n,med}` (same unit as ``r``).
        sigma : float
            Geometric standard deviation :math:`\sigma` (natural log).
        '''
        mu = np.log(rn_med)
        return (np.exp(-(np.log(r) - mu) ** 2 / (2 * sigma ** 2))
                / (r * sigma * np.sqrt(2 * np.pi)))

    def microplastic_AF2021(self, r, gamma=.5):
        r'''Size distribution of microplastics from Aoki and Furue (2021).

        .. math:: n(r) \propto \frac{d^{-4}}{\exp\left[1/(\gamma d)\right] - 1},
                  \qquad d = 2r

        normalized to :math:`\int n(r)\,dr = 1`
        (https://doi.org/10.1371/journal.pone.0259781).
        '''
        d = 2 * r
        psd = d ** -4 / (np.exp(1 / (gamma * d)) - 1)
        norm = trapezoid(psd, r)
        return psd/norm

    def power_law_junge(self,r, slope=-3.5, rmin=0.03, rmax=100):
        r'''Junge (power-law) distribution normalized to :math:`\int n(r)\,dr = 1`.

        .. math:: n(r) \propto r^{\xi} \text{ for } r_{min} \le r \le r_{max},
                  \quad 0 \text{ elsewhere}

        with :math:`\xi` the ``slope``.
        '''
        psd = np.array(r ** slope)
        psd[r < rmin] = 0.
        psd[r > rmax] = 0.
        return psd / trapezoid(psd, r)

    def modif_power_law(self,r, slope=-3.5, rmin=0.03, rmax=100):
        r'''Modified power law normalized to :math:`\int n(r)\,dr = 1`.

        .. math:: n(r) \propto (r/r_{min})^{\xi} \text{ for } r_{min} \le r \le r_{max},
                  \quad 1 \text{ for } r < r_{min}, \quad 0 \text{ for } r > r_{max}
        '''
        psd = np.array((r / rmin) ** slope)
        psd[r < rmin] = 1
        psd[r > rmax] = 0.
        return psd / trapezoid(psd, r)

    def rmod2rmed(self,rmod, sig):
        r'''Log-normal median radius from the modal radius:
        :math:`\ln r_{med} = \ln r_{mod} + \sigma^2`.'''
        return np.exp(np.log(rmod) + sig ** 2)

    def rmed2rmod(self,rmed, sig):
        r'''Log-normal modal radius from the median radius:
        :math:`\ln r_{mod} = \ln r_{med} - \sigma^2`.'''
        return np.exp(np.log(rmed) - sig ** 2)

    def muv2mun(self,muv, sig):
        r'''Log of the number median radius from that of the volume distribution:
        :math:`\mu_n = \mu_v - 3\sigma^2`.'''
        return muv - 3 * sig ** 2

    def mun2muv(self,mun, sig):
        r'''Log of the volume median radius from that of the number distribution:
        :math:`\mu_v = \mu_n + 3\sigma^2`.'''
        return mun + 3 * sig ** 2

    def rvmod2rnmed(self,rv_mod, sig):
        r'''Number median radius from the modal radius of the volume distribution:
        :math:`r_{n,med} = r_{v,mod}\, e^{-2\sigma^2}`.'''
        return rv_mod * np.exp(-2 * sig ** 2)

    def rnmed2rvmed(self,rn_med, sig):
        r'''Volume median radius from the number median radius:
        :math:`\ln r_{v,med} = \ln r_{n,med} + 3\sigma^2`.'''
        return np.exp(np.log(rn_med) + 3 * sig ** 2)

    def arr_format(self,arr, fmt="{:0.1f}"):
        return [fmt.format(x) for x in arr]


class SizeParam:
    r'''Size distribution and geometrical parameters (Hansen and Travis, 1974).

    .. math::

        r_{mean} = \frac{\int r\, n(r)\, dr}{\int n(r)\, dr}, \qquad
        G = \int \pi r^2 n(r)\, dr, \qquad
        r_{eff} = \frac{1}{G}\int r\, \pi r^2 n(r)\, dr, \qquad
        v_{eff} = \frac{1}{r_{eff}^2 G}\int (r - r_{eff})^2\, \pi r^2 n(r)\, dr

    Parameters
    ----------
    radius : numpy.ndarray
        Radius :math:`r`.
    psd : numpy.ndarray
        Number size distribution :math:`n(r)`, modified in place (NaN set to 0
        for the integration, then 0 set to NaN).
    '''

    def __init__(self, radius, psd):
        self.psd = psd
        self.psd[np.isnan(self.psd)] = 0
        self.radius = radius
        self.rmean = self.integr() / self.integr(0)
        self.G = self.moment(2)
        self.reff = self.moment(3) / self.G
        self.veff = self.variance(self.G, self.reff)
        self.sigeff = self.veff ** 0.5
        self.psd[self.psd == 0] = np.nan

    def to_dict(self):
        return {
            'G': self.G,
            'rmean': self.rmean,
            'reff': self.reff,
            'veff': self.veff,
            'sigeff': self.sigeff,
        }

    def to_annotation_r(self):
        return '$r_{mean}=$' + '{:.2e}'.format(self.rmean) + ' $\\mu m$\n $r_{eff}=$' + '{:.2e}'.format(
            self.reff) + r' $\mu m$'

    def to_annotation(self):
        return r'$r_{mean}=$' + '{:.3f}'.format(self.rmean) + '$\\mu m$\n$ \\ r_{eff}=$' + '{:.3f}'.format(
            self.reff) + '$\\mu m$\n$ \\sigma^2_{eff}=$' + '{:.3f}'.format(self.veff) + r'$\mu m^2$'

    def integr(self, order=1):
        r''':math:`\int r^{order}\, n(r)\, dr`'''
        _f = self.psd * self.radius ** order
        return trapezoid(_f, self.radius)

    def moment(self, order=1):
        r''':math:`\int \pi r^{order}\, n(r)\, dr`'''
        _f = self.psd * np.pi * self.radius ** order
        return trapezoid(_f, self.radius)

    def variance(self, G, reff):
        r'''Effective variance
        :math:`v_{eff} = \frac{1}{r_{eff}^2 G}\int (r - r_{eff})^2\, \pi r^2 n(r)\, dr`'''
        _f = self.psd * np.pi * ((self.radius - reff) * self.radius) ** 2
        return trapezoid(_f, self.radius) / (reff ** 2 * G)
