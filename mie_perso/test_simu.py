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

pmie = mie_multiprocess.Processor()

size_param = psd.SizeParam
psd = psd.PSD()


theta = np.linspace(0, np.pi, 3600)
#test x max
u = np.cos(theta)
m_=complex(1.15,0e-3)
wl=500
Qeff=[]
x = np.logspace(-1, np.log10(40000), 501)




mr_list=[1.05,1.07,1.08,1.20]
x = np.logspace(-1, np.log10(40000), 501)

mr_list = np.arange(1.005, 1.22, 0.005)
mi = 0
Qeff = []
wl = 500
for x_ in x:
    p=[]
    for mr in mr_list:
        m=complex(mr, mi)
        diameter = wl *  x_ /np.pi
        p.append([x_,mr, mi])
    pool = multiprocessing.Pool(processes=38)
    res = pool.map(pmie.Qeff_compute, p)

    Qeff.append(res)


xQeff=pd.DataFrame(np.concatenate(Qeff),
                   columns=['x','nr','ni','Qext', 'Qsca', 'Qabs', 'g', 'Qpr', 'Qback', 'Qratio']
                   ).set_index(['x','nr','ni']).to_xarray()
ofile = 'mie_perso/data/Qeff_mie_microplastic_v4.nc'
xQeff.to_netcdf(ofile)

plt.figure()
xQeff.Qsca.plot(hue='mr')

plt.semilogx()

m = complex(1.1, 0)
theta = np.linspace(0, np.pi, 3600)
x=np.array([100])

mueller = pmie.ScatMat_mp(m, x, theta)
wl=500
diameter = wl *  x /np.pi
Qeff = ps.MieQ(m, wl, diameter,asDict=True)

S11 = mueller.S11
S12 = mueller.S12 / S11
S33 = mueller.S33 / S11
S34 = mueller.S34 / S11

P11 = wl**2/np.pi * S11
Cscat = trapezoid(P11 * np.sin(theta), theta) / 2
Csca = Qeff['Qsca']*np.pi*diameter**2/4
print(Cscat/Csca)
#3.73311362e+08
#8.39950564e+08
P11

