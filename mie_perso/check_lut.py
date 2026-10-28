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

from mie_perso import psd, mie_multiprocess


size_param = psd.SizeParam
psd = psd.PSD()


ofile = 'mie_perso/data/mueller_mie_microplastic.nc'
ofile = 'mie_perso/data/mueller_mie_microplastic1.050+0.000j.nc'

if os.path.exists(ofile):
    mueller = xr.open_dataset(ofile)
else:
    print('error in file name, try again!')

# -------------------------------------
# Plot scattering functions
# for a series of size parameters x (2pi r / lambda)
# -------------------------------------

theta = mueller.theta
ang = theta * 180. / np.pi
Nx = len(mueller.x)
inc = int(Nx/40)
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'black',
                                                     'orangered', "firebrick", "darkred", 'yellowgreen', 'forestgreen'])

# cmap = plt.cm.bone_r
# norm = mpl.colors.Normalize(vmin=0, vmax=500)
try:
    mueller_ = mueller.isel(x=np.arange(0,Nx,inc)).isel(nr=0)
except:
    mueller_ = mueller.isel(x=np.arange(0,Nx,inc))


norm = mpl.colors.LogNorm(vmin=mueller_.x.min(), vmax=mueller_.x.max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.985, left=0.1, right=0.98,
                    hspace=0.01, wspace=0.025)

for x_, m_ in list(mueller_.groupby('x')):
    axs[0].plot(ang, m_.S11, color=cmap(norm(x_)), alpha=0.6, lw=1)
    axs[1].plot(ang, m_.S11, color=cmap(norm(x_)), alpha=0.6, lw=1)
axs[1].minorticks_on()
axs[0].semilogy()
axs[1].loglog()
axs[0].set_ylabel('$S_{11}$')
axs[0].set_xlabel('$Scattering\ angle\ (deg)$')
axs[1].set_xlabel('$Scattering\ angle\ (deg)$')
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top', format=mpl.ticker.ScalarFormatter())
cb.set_label('x-parameter', fontsize=22)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True)
fig.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.9,
                    hspace=0.1, wspace=0.05)

for x_, m_ in list(mueller_.groupby('x')):
    axs[0, 0].plot(ang, m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
    axs[0, 1].plot(ang, m_.S12 / m_.S11, color=cmap(norm(x_)), alpha=0.2, lw=1)
    axs[1, 0].plot(ang, m_.S33 / m_.S11, color=cmap(norm(x_)), alpha=0.2, lw=1)
    axs[1, 1].plot(ang, m_.S34 / m_.S11, color=cmap(norm(x_)), alpha=0.2, lw=1)
axs[0, 0].semilogy()
for i in [0, 1]:
    axs[i, 1].yaxis.set_label_position("right")
    axs[i, 1].yaxis.tick_right()
    for j in [0, 1]:
        axs[i, j].minorticks_on()
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top', format=mpl.ticker.ScalarFormatter())
cb.set_label('x-parameter', fontsize=20)
#
# S11tot = np.sum(S11, axis=0)
# axs[0, 0].plot(theta, S11tot, 'k', lw=2, label='tot')
# axs[0, 1].plot(theta, np.sum(S12, axis=0) / S11tot, 'k', lw=2)
# axs[1, 0].plot(theta, np.sum(S33, axis=0) / S11tot, 'k', lw=2)
# axs[1, 1].plot(theta, np.sum(S34, axis=0) / S11tot, 'k', lw=2)

axs[0, 0].set_ylabel('$S_{11}$')
axs[0, 1].set_ylabel('$S_{12}$')
axs[1, 0].set_ylabel('$S_{33}$')
axs[1, 1].set_ylabel('$S_{34}$')
axs[1, 0].set_xlabel('$Scattering\ angle\ (deg)$')
axs[1, 1].set_xlabel('$Scattering\ angle\ (deg)$')

axs[1, 0].set_ylim([-1.05, 1.05])
axs[1, 1].set_ylim([-1.05, 1.05])
axs[0, 1].set_ylim([-1.05, 1.05])
