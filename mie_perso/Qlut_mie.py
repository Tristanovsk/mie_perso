import os

import numpy as np
import pandas as pd
import xarray as xr
import dask
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

from mie_perso import psd, mie_multiprocess, size_average

pmie = mie_multiprocess.Processor()
size_param = psd.SizeParam
psd = psd.PSD()

# -------------------------------------
# Generate efficiency factors for
# size parameters x = np.pi * diameter / wavelength (unitless)
# -------------------------------------

xs = np.linspace(1, 2000, 2000)
xs = np.logspace(-3,1,1000)
mrs = np.linspace(1.12, 1.23, 11)
mis = np.linspace(0, 0.05, 201)
mis = np.logspace(-5, -1, 100)
#xs = np.linspace(1, 4000, 4)
#mrs = np.linspace(1.01, 1.25, 2)
#mis = np.linspace(0, 0.1, 2)
#


ofile = '/data/vrtc/xlut/Q_mie_sed_small_size.nc'


if os.path.exists(ofile):
    xQeff = xr.open_dataset(ofile)
else:
    p = []
    for mr in mrs:
        for mi in mis:
            for x in xs:
                p.append([x, mr, mi])
    with multiprocessing.Pool(processes=pmie.processes) as pool:
        res = pool.map(pmie.Qeff_compute, p)

    xQeff = pd.DataFrame(res,
                         columns=['x', 'mr', 'mi', 'Qext', 'Qsca', 'Qabs', 'g', 'Qpr', 'Qback', 'Qratio']
                         ).set_index(['x', 'mr', 'mi']).to_xarray()

    xQeff.to_netcdf(ofile)


# -------------------------------------
# quick look at the look-up table
# (size-distribution integrals: see plastic_mie.py and mie_perso.size_average)
# -------------------------------------
mr_ = xQeff.mr.values[len(xQeff.mr) // 2]
mi_sel = xQeff.mi.values[::20]
cmap = plt.cm.viridis
norm = mpl.colors.LogNorm(vmin=mi_sel.min(), vmax=mi_sel.max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.8, left=0.07, right=0.98, wspace=0.3)
for mi in mi_sel:
    Q = xQeff.sel(mr=mr_, mi=mi)
    for ax, var in zip(axs, ['Qsca', 'Qabs', 'Qback']):
        ax.plot(Q.x, Q[var], color=cmap(norm(mi)), lw=1.2)
for ax, var in zip(axs, ['Qsca', 'Qabs', 'Qback']):
    ax.loglog()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$' + var[0] + '_{' + var[1:] + '}$')
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
cb.set_label('$n_i$ ($n_r$ = {:.3f})'.format(mr_), fontsize=20)
plt.show()
