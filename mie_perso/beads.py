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

from mie_perso import psd, mie_multiprocess
p = mie_multiprocess.Processor()
size_param = psd.SizeParam
psd = psd.PSD()


# -------------------------------------
# Generate mueller matrices for a series
# of size parameters x = np.pi * diameter / wavelength (unitless)
# -------------------------------------
wl = 515
nMedium = 1.3199+6878/wl**2-1.132e9/wl**4+1.11e14/wl**6
wl_medium=wl/nMedium
npolystyrene = 1.60
m = npolystyrene / nMedium - 0.000j

theta = np.linspace(0, np.pi, 3600)
x = np.logspace(np.log10(1), 2, 1001)
ofile = 'mie_perso/data/mueller_mie_' + format(m,'1.3f') + '_t3600.nc'

if os.path.exists(ofile):
    mueller = xr.open_dataset(ofile)
else:
    mueller = p.ScatMat_mp(m, x, theta)
    mueller.to_netcdf(ofile)

# -------------------------------------
# convert mueller matrices for a series
# of diameters for a given couple of
# wavelength (in vacuum) and refractive index (in medium)
# for a medium of refractive index nMedium
# -------------------------------------
theta = mueller.theta
ang = theta * 180. / np.pi
Ntheta = len(mueller.theta)

m = complex(mueller.nr, mueller.ni)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)
fig.subplots_adjust(bottom=0.115, top=0.962, left=0.086, right=0.98,
                    hspace=0.01, wspace=0.01)

m_vacuum = m * nMedium
for wavelength in [400, 500, 600, 700]:
    wl_medium = wavelength / nMedium

    dpnm = mueller.x * wl_medium / np.pi
    dp = dpnm / 1000

    rn_med = 1.5
    CV = 0.01
    sig = rn_med * CV
    rv_med = psd.rnmed2rvmed(rn_med, sig)

    ndp = psd.lognorm(dp / 2, rn_med=rn_med, sigma=sig)
    # ndp = modif_power_law(dp / 2, slope=-slope, rmin=rmin, rmax=rmax)
    # convert to xarray
    ndp = dp.copy(data=ndp)

    # ndp = psd #[:-1]*np.diff(dp/2)
    S11, S12, S33, S34 = np.zeros(Ntheta), np.zeros(Ntheta), np.zeros(Ntheta), np.zeros(Ntheta)

    # aSDn = np.pi*((dp/2)**2)*ndp
    aSDn = ndp
    S11 = np.trapz(mueller.S11 * aSDn, dp, axis=0)
    S12 = np.trapz(mueller.S12 * aSDn, dp, axis=0)
    S33 = np.trapz(mueller.S33 * aSDn, dp, axis=0)
    S34 = np.trapz(mueller.S34 * aSDn, dp, axis=0)

    norm = trapz(S11 * np.sin(theta), theta) / 2
    axs[0, 0].plot(ang, S11 / norm, lw=1, label='$wl_0=$'+str(wl)+'$wl_w=$'+str(wl_medium))
    axs[0, 0].semilogy()
    axs[0, 1].plot(ang, S12 / S11, lw=1)
    axs[1, 0].plot(ang, S33 / S11, lw=1)
    axs[1, 1].plot(ang, S34 / S11, lw=1)

axs[0, 0].set_ylabel('S11')
axs[1, 0].set_xlabel('Scattering angle (deg)')
axs[1, 1].set_xlabel('Scattering angle (deg)')

axs[1, 0].set_ylim([-1.05, 1.05])
axs[1, 1].set_ylim([-1.05, 1.05])
axs[0, 1].set_ylim([-1.05, 1.05])

plt.show()

plt.savefig('mie_perso/fig/beads_radius' + str(rn_med) + '_sig'+str(sig)+'_m' +format(m,'1.3f')+'.png', dpi=300)
