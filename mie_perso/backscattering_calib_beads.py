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

mpl.use('tkagg')
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
opj = os.path.join
figdir = 'mie_perso/fig/roettgers'

# -------------------------------------
# Generate mueller matrices for a series
# of size parameters x = np.pi * diameter / wavelength (unitless)
# -------------------------------------
# list of refractive index in medium
mi = 0.
m_list = [complex(1.02, 0), complex(1.04, 0), complex(1.06, 0)]

theta = np.linspace(0, np.pi, 3600)
x = np.linspace(1, 40000, 1001)
x = np.logspace(-1, np.log10(40000), 501)

x = np.logspace(-1, np.log10(75), 1101)

# test x for given wl and psd:
nMedium = 1.334
wl = 600
wl_medium = wl / nMedium
radius_nm = x * wl_medium / (2 * np.pi)
radius_mic = 1e-3 * radius_nm
radius_mm = 1e-3 * radius_mic

ofile = 'data/mueller_mie_formazin_v4.nc'
qfile = 'data/Qeff_mie_formazin_v4.nc'
if os.path.exists(ofile):
    mueller = xr.open_dataset(ofile, chunks={'nr': 1})
    xQeff = xr.open_dataset(qfile)
else:
    m_ = []
    mi = 0.
    mr_list = np.arange(1.005, 1.25, 0.005)
    # generate scattering matrix
    for mr in mr_list:
        m = complex(mr, mi)
        res = pmie.ScatMat_mp(m, x, theta)
        ofile_ = 'data/mueller_mie_formazin' + format(m, '1.3f') + '_v4.nc'
        res.to_netcdf(ofile_)
        m_.append(res)
    mueller = xr.concat(m_, "nr")
    mueller.to_netcdf(ofile)

    # generate Qeff
    # use of fake wavelength and diameter based on x
    Qeff = []
    wl = 500

    for x_ in x:
        p = []
        for mr in mr_list:
            m = complex(mr, mi)
            diameter = wl * x_ / np.pi
            p.append([x_, mr, mi])
        with multiprocessing.Pool(processes=pmie.processes) as pool:
            res = pool.map(pmie.Qeff_compute, p)

        Qeff.append(res)

    xQeff = pd.DataFrame(np.concatenate(Qeff),
                         columns=['x', 'nr', 'ni', 'Qext', 'Qsca', 'Qabs', 'g', 'Qpr', 'Qback', 'Qratio']
                         ).set_index(['x', 'nr', 'ni']).to_xarray()

    xQeff.to_netcdf(qfile)

# xQeff=xQeff.rename({'mr':'nr','mi':'ni'})
mueller = mueller.dropna('x')
mueller = mueller.chunk(chunks={'nr': 1})
x = mueller.x

radius_nm = x * wl_medium / (2 * np.pi)
radius_mic = 1e-3 * radius_nm
radius_mm = 1e-3 * radius_mic
radius = radius_mm

# -------------------------------------
# convert mueller matrices for a series
# of diameters for a given couple of
# wavelength (in vacuum) and refractive index (in medium)
# for a medium of refractive index nMedium
# -------------------------------------

nMedium = 1.334
theta = mueller.theta
ang = theta * 180. / np.pi
Ntheta = len(mueller.theta)
# test size ditribution vs size parameter

size_class = 'small'
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', 'lightskyblue',
                                                     'orangered', "firebrick", "darkred",
                                                     'yellowgreen', 'forestgreen'])

norm = mpl.colors.LogNorm(vmin=1.05, vmax=1.2)
norm = mpl.colors.Normalize(vmin=1.0, vmax=1.08)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
sigma = 1.0
lw = 1.2
alpha = 0.85
# set wavelength as xarray for further computation
wls = np.arange(300, 850, 1)  # *1e-3
wls = xr.DataArray(wls, coords={'wavelength': wls})

# for igamma, gamma in enumerate([ 0.5, 1. , 2.,6.]):
gamma = 2.

# -----------------------------------
# First assumption:
# simple test for monosized beads
# -----------------------------------
npolystyrene = 1.60
n_w = 1.334
m = npolystyrene / nMedium

sample = 'Beads_3um'
# put a fake number concentration (in number / micron**3)
concentration = 0.098
radius_nm = 1500
radius_mic = radius_nm * 1e3
wl0 = wls
wl0_mic = wl0 * 10 ** -3

wl_water = wl0 / n_w
x_ = 2 * np.pi * radius_nm / wl_water

b_coef = concentration * xQeff.Qback.isel(nr=range(-18, -5), ni=0).interp(x=x_) * np.pi * radius_mic ** 2
b_coef_beads = b_coef.interp(nr=m)

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen'])

norm = mpl.colors.Normalize(vmin=1.16, vmax=1.22)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)

# plt.legend()
mono_index = False
suff = ''
if mono_index:
    suff = '_mono_index'
    nr = 1.20
    arr = b_coef.interp(nr=1.20)
    axs.plot(arr.wavelength, arr, color='gray', lw=1.5, label='Mie theory')
else:

    for nr, arr in b_coef.sel(nr=b_coef.nr[0:-1:2]).groupby('nr'):
        print(nr)
        axs.plot(arr.wavelength, arr, color=cmap(norm(nr)), alpha=0.6, lw=1.5)

axs.set_title('')
axs.set_ylabel('$Scattering\ coef.\ (m^{-1})$')
axs.set_xlabel('$Wavelength\ (nm)$')
plt.legend()
if not mono_index:
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
    cb.set_label('$n_r$', fontsize=22)

plt.suptitle(sample)
plt.savefig(opj(figdir, 'scat_coef_beads_first_assumption_' + sample + suff + '.png'), dpi=300)

# -----------------------------------
# Second assumption:
# consideration of spectral refractive index
# -----------------------------------

# sample='Beads_300nm'
# concentration = 75
# radius_nm=150

radius_mic = 1e-3 * radius_nm

CV = 0.0051
rn_med = radius_mic
sig = rn_med * CV
rv_med = psd.rnmed2rvmed(rn_med, sig)
extent = 0.1
radius_ = np.linspace(radius_mic * (1 - extent), radius_mic * (1 + extent), 150)


def n_polystyrene(wl0_mic):
    '''
    Polystyrene refractive index from Sultanova et al. 2009
    :param wl0_mic: vacuum wavelength in micrometers
    :return:
    '''
    return np.sqrt(1 + 1.4435 * wl0_mic ** 2 / (wl0_mic ** 2 - 0.020216))


def n_water(wl0_nm):
    '''
    Water refractive index
    :param wl0_nm: vacuum wavelength in nanometers
    :return:
    '''
    return 1.3199 + 6878 / wl0_nm ** 2 - 1.132e9 / wl0_nm ** 4 + 1.11e14 / wl0_nm ** 6


def n_water_perso(wl0_nm, water_temp=20, water_sal=0):
    '''
    load refractive index from Harmel et al., 2018
    :param wl0_nm:
    :param water_temp:
    :param water_sal:
    :return:
    '''

    import refractive_index.water_refractive_index.auxdata as water
    wri = water.water_refractive_index()
    # water_temp,water_sal = 10,30
    return wri.n_harmel(wl0_nm, water_temp, water_sal)


# -------------------------------------
# spectral variation of the refractive index:
# -------------------------------------
suff = ''
refrac_water_perso = True
if refrac_water_perso:
    suff = '_nwperso'
    wl = wl0.data.astype(float)
    n_w = n_water_perso(wl, water_temp=20, water_sal=0)
else:
    n_w = n_water(wl0)

n_p = n_polystyrene(wl0_mic)
m = n_p / n_w

plt.figure()
plt.plot(wl0, n_w, label='n_w in vacuum')
plt.plot(wl0, n_p, label='n_polystyr. in vacuum')
plt.plot(wl0, m, label='n_polystyr. in water')
plt.legend()
plt.savefig(opj(figdir, 'polystyrene_refractive_index.png'), dpi=300)

wl_water = wl0 / n_w
for diam_mic in [0.3, 0.7, 1, 3]:
    radius_mic = diam_mic / 2.
    rn_med = radius_mic
    radius_nm = radius_mic * 1e3
    sample = 'polystyrene_beads_{:.3f}micron'.format(diam_mic)
    print(sample)

    # set radius for size distribution computation (ndp)
    radius_ = np.linspace(radius_mic * (1 - extent), radius_mic * (1 + extent), 150)
    ndp = psd.lognorm(radius_, rn_med=rn_med, sigma=sig)

    # set size parameter
    x_ = 2 * np.pi * radius_nm / wl_water

    b_coef = concentration * xQeff.Qback.isel(nr=range(-19, 0), ni=0).interp(x=x_) * np.pi * radius_mic ** 2
    b_coef_beads = b_coef.interp(nr=m)

    # -------------------------------------
    # size distribution:
    # -------------------------------------

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)
    arr = []
    for ir, radius in enumerate(radius_):
        wgt = ndp[ir]
        rad_nm = radius * 1000
        x_ = 2 * np.pi * rad_nm / wl_water
        alpha = (wgt / ndp.max()) ** 0.15
        b_coef_tmp = concentration * xQeff.Qback.isel(nr=range(-19, 0), ni=0).interp(x=x_) * np.pi * radius_mic ** 2
        b_coef_ = b_coef_tmp.interp(nr=m)

        b_coef_.plot.line(x='wavelength', color='black', alpha=alpha, ls='-', lw=1.3, label='computed', ax=axs)
        arr.append(b_coef_.assign_coords(radius=radius) * wgt)

    b_coef_ = xr.concat(arr, dim='radius')
    b_coef_int = b_coef_.integrate('radius').compute()
    b_coef_int.plot(x='wavelength')

    # -------------------------------------
    # PLOTTING:
    # -------------------------------------
    cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                        ['navy', "blue", 'lightskyblue',
                                                         'gray', 'yellowgreen', 'forestgreen'])

    norm = mpl.colors.Normalize(vmin=1.18, vmax=1.23)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)
    axs.minorticks_on()
    # plt.legend()
    for nr, arr in b_coef.sel(nr=b_coef.nr[:]).groupby('nr'):
        print(nr)
        axs.plot(arr.wavelength, arr, color=cmap(norm(nr)), alpha=0.6, lw=1.5)

    b_coef_beads.plot.line(x='wavelength', color='black', ls=':', lw=1.8, label='computed', ax=axs)
    b_coef_int.plot.line(x='wavelength', color='black', ls='-', lw=2.3, label='computed with size ditrib.', ax=axs)

    axs.set_title('')
    axs.set_ylabel('$Backscattering\ coef.\ (m^{-1})$')
    axs.set_xlabel('$Wavelength\ (nm)$')
    plt.legend()
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
    cb.set_label('$n_r$', fontsize=22)
    plt.suptitle(sample)
    plt.savefig(opj(figdir, 'scat_coef_beads_' + sample + suff + '.png'), dpi=300)


#------------------------------------------------------_
#------------------------------------------------------_
# Now from the scattering function at one given angle
#------------------------------------------------------_
#------------------------------------------------------_
# -------------------------------------
# get the angle(s) you want:
angle=114
theta_rad = np.radians(angle)
xarr = mueller.S11.sel(theta=theta_rad,method="nearest")
suff = '_nwperso'

angle=[110,111,112,113,114,115,116,117,118,119,120]

ang_min,ang_max=np.radians(95),np.radians(135)
theta_rad = np.radians(angle)
xarr = mueller.S11.sel(theta=slice(ang_min,ang_max)) # .integrate('theta')
angles = np.degrees(xarr.theta)
dist = psd.norm(angles,mu=114,sigma=5)
plt.figure()
plt.plot(angles,dist)
plt.xlabel('Scattering angle (deg.)')
plt.ylabel('Distribution')
plt.savefig(opj(figdir,'distribution_angle_bb_sensor.png'))
# convolution with norm distribution: weighted integral over theta in one pass over the LUT
xarr = size_average.size_average(xarr, dist * size_average.trapezoid_weights(xarr.theta), dim='theta').compute()

suff='_nwperso_integr'

# -------------------------------------
# spectral variation of the refractive index:
# -------------------------------------
wl_water = wl0 / n_w
for diam_mic in [0.3, 0.7, 1, 3]:
    radius_mic = diam_mic / 2.
    rn_med = radius_mic
    radius_nm = radius_mic * 1e3
    sample = 'polystyrene_beads_{:.3f}micron'.format(diam_mic)
    print(sample)

    # set radius for size distribution computation (ndp)
    radius_ = np.linspace(radius_mic * (1 - extent), radius_mic * (1 + extent), 150)
    ndp = psd.lognorm(radius_, rn_med=rn_med, sigma=sig)

    # set size parameter
    x_ = 2 * np.pi * radius_nm / wl_water

    b_coef = concentration * xarr.isel(nr=range(-19, 0)).interp(x=x_) * np.pi * radius_mic ** 2
    b_coef_beads = b_coef.interp(nr=m)

    # -------------------------------------
    # size distribution:
    # -------------------------------------

    #fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)
    arr = []
    for ir, radius in enumerate(radius_):
        wgt = ndp[ir]
        rad_nm = radius * 1000
        x_ = 2 * np.pi * rad_nm / wl_water
        alpha = (wgt / ndp.max()) ** 0.15
        b_coef_tmp = concentration * xarr.isel(nr=range(-19, 0)).interp(x=x_) * np.pi * radius_mic ** 2
        b_coef_ = b_coef_tmp.interp(nr=m)

        #b_coef_.plot.line(x='wavelength', color='black', alpha=alpha, ls='-', lw=1.3, label='computed', ax=axs)
        arr.append(b_coef_.assign_coords(radius=radius) * wgt)

    b_coef_ = xr.concat(arr, dim='radius')
    b_coef_int = b_coef_.integrate('radius').compute()
    #b_coef_int.plot(x='wavelength')

    # -------------------------------------
    # PLOTTING:
    # -------------------------------------
    cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                        ['navy', "blue", 'lightskyblue',
                                                         'gray', 'yellowgreen', 'forestgreen'])

    norm = mpl.colors.Normalize(vmin=1.18, vmax=1.23)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)
    axs.minorticks_on()
    # plt.legend()
    for nr, arr in b_coef.sel(nr=b_coef.nr[:]).groupby('nr'):
        print(nr)
        axs.plot(arr.wavelength, arr, color=cmap(norm(nr)), alpha=0.6, lw=1.5)

    b_coef_beads.plot.line(x='wavelength', color='black', ls=':', lw=1.8, label='computed', ax=axs)
    b_coef_int.plot.line(x='wavelength', color='black', ls='-', lw=2.3, label='computed with size ditrib.', ax=axs)

    axs.set_title('')
    axs.set_ylabel('$Backscattering\ coef.\ (m^{-1})$')
    axs.set_xlabel('$Wavelength\ (nm)$')
    plt.legend()
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
    cb.set_label('$n_r$', fontsize=22)
    plt.suptitle(sample)
    plt.savefig(opj(figdir, 'scat_function_{:.0f}_beads_'.format(angle) + sample + suff + '.png'), dpi=300)









# ----------------------------------------------
# effect of temperature or salinity
# ----------------------------------------------

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen', 'orange', 'red'])
norm = mpl.colors.Normalize(vmin=5, vmax=30)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)

att_coef[sample].plot(x='wavelength', color='red', lw=3.5, label='measured', ax=axs)
# plt.legend()
for tempC in [0, 10, 20, 30, 40]:
    wl = wl0.data.astype(float)
    n_w = n_water_perso(wl, water_temp=20, water_sal=tempC)

    n_p = n_polystyrene(wl0_mic)
    m = n_p / n_w
    arr = []
    for ir, radius in enumerate(radius_):
        wgt = ndp[ir]
        rad_mic = radius * 1000
        x_ = 2 * np.pi * rad_mic / wl_water
        alpha = (wgt / ndp.max()) ** 0.15
        b_coef = concentration * xQeff.Qsca.isel(nr=range(-19, 0), ni=0).interp(x=x_) * np.pi * radius_mic ** 2
        b_coef_ = b_coef.interp(nr=m)

        arr.append(b_coef_.assign_coords(radius=radius) * wgt)

    b_coef_ = xr.concat(arr, dim='radius')
    b_coef_int = b_coef_.integrate('radius').compute()

    axs.plot(b_coef_int.wavelength, b_coef_int, color=cmap(norm(tempC)), alpha=0.6, lw=2)

axs.set_title('')
axs.set_ylabel('$Scattering\ coef.\ (m^{-1})$')
axs.set_xlabel('$Wavelength\ (nm)$')
plt.legend()
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
# cb.set_label('$Water\ temperature\ (\deg C)$', fontsize=22)
cb.set_label('$Water\ salinity\ (psu)$', fontsize=22)

plt.suptitle(sample)
plt.savefig(opj(figdir, 'scat_coef_beads_' + sample + suff + '_water_salinity.png'), dpi=300)

# x_ = 2*np.pi *radius_nm /(att_coef.wavelength/ 1.334)
# b_coef = concentration * xQeff.Qsca.isel(nr=[-6,-5],ni=0).interp(x=x_) *np.pi*radius_mic**2
# b_coef.plot.line(x='wavelength')
