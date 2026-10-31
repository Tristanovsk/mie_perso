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
opj = os.path.join
figdir='mie_perso/fig/formazin'

# -------------------------------------------------
# open data file nd load spectral attenuation coef
# and mueller matrix terms
# -------------------------------------------------
att_file = 'data/lab/lov_attenuation_feb2015.xlsx'
# mueller_file=
att_coef = pd.read_excel(att_file, index_col=0, sheet_name='corrected_c').to_xarray()
samples = ['Beads_300nm', 'Beads_3um', 'Formazin_2NTU', 'Formazin_4NTU']
# plot raw data:
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.925, left=0.085, right=0.98,
                    hspace=0.01, wspace=0.2)

for i,sample in enumerate(samples[:2]):
    att_coef[sample].plot(ax=axs[i])
    axs[i].set_title(sample)
    axs[i].set_xlabel('$Wavelength\ (nm)$')
    axs[i].set_ylabel('$Attenuation\ coef.\ (m^{-1})$')

plt.savefig(opj(figdir,'att_coef_beads.png'), dpi=300)




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

fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(21, 15), sharex=True, sharey='row')
fig.subplots_adjust(bottom=0.115, top=0.98, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.05)
norm = mpl.colors.LogNorm(vmin=1.05, vmax=1.2)
norm = mpl.colors.Normalize(vmin=1.0, vmax=1.08)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
sigma = 1.0
lw = 1.2
alpha = 0.85

# for igamma, gamma in enumerate([ 0.5, 1. , 2.,6.]):
gamma = 2.
# mean scattering cross section (micron**2) of a lognormal population (rn_med in micron)
weights = size_average.psd_weights(
    xQeff.x, att_coef.wavelength.values * 1e-3, lambda r: psd.lognorm(r, rn_med=gamma, sigma=sigma),
    n_medium=nMedium).assign_coords(wavelength=att_coef.wavelength.values)
Csca_hyp = size_average.integrate_cross_section(
    xQeff.isel(nr=[1, 4, 8, 12], ni=0), weights, variables=('Qsca',)).Csca
Csca_hyp.plot.line(x='wavelength')

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
radius_mic = 1e-3 * radius_nm

wl0 = att_coef.wavelength
wl0_mic = wl0 * 10 ** -3

wl_water = wl0 / n_w
x_ = 2 * np.pi * radius_nm / wl_water

b_coef = concentration * xQeff.Qsca.isel(nr=range(-18, -5), ni=0).interp(x=x_) * np.pi * radius_mic ** 2
b_coef_beads = b_coef.interp(nr=m)


cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen'])

norm = mpl.colors.Normalize(vmin=1.16, vmax=1.22)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)

att_coef[sample].plot(x='wavelength', color='red', lw=3.5, label='measured', ax=axs)
# plt.legend()
mono_index=False
suff=''
if mono_index:
    suff='_mono_index'
    nr=1.20
    arr = b_coef.interp(nr=1.20)
    axs.plot(arr.wavelength, arr, color='gray', lw=1.5,label='Mie theory')
else:

    for nr in b_coef.nr.values[0:-1:2]:
        axs.plot(b_coef.wavelength, b_coef.sel(nr=nr), color=cmap(norm(nr)), alpha=0.6, lw=1.5)

axs.set_title('')
axs.set_ylabel('$Scattering\ coef.\ (m^{-1})$')
axs.set_xlabel('$Wavelength\ (nm)$')
plt.legend()
if not mono_index:
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
    cb.set_label('$n_r$', fontsize=22)
plt.suptitle(sample)
plt.savefig(opj(figdir,'scat_coef_beads_first_assumption_' + sample + suff+'.png'), dpi=300)



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
ndp = psd.lognorm(radius_, rn_med=rn_med, sigma=sig)
plt.plot(radius_, ndp)


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

def n_water_perso(wl0_nm,water_temp=20,water_sal=0):
    '''
    load refractive index from Harmel et al., 2018
    :param wl0_nm:
    :param water_temp:
    :param water_sal:
    :return:
    '''

    import refractive_index.water_refractive_index.auxdata as water
    wri = water.water_refractive_index()
    #water_temp,water_sal = 10,30
    return wri.n_harmel(wl0_nm,water_temp,water_sal)

# -------------------------------------
# spectral variation of the refractive index:
# -------------------------------------
suff=''
refrac_water_perso = True
if refrac_water_perso:
    suff='_nwperso'
    wl = wl0.data.astype(float)
    n_w = n_water_perso(wl,water_temp=20,water_sal=0)
else:
    n_w = n_water(wl0)

n_p = n_polystyrene(wl0_mic)
m = n_p / n_w

plt.figure()
plt.plot(wl0,n_w,label='n_w in vacuum')
plt.plot(wl0,n_p,label='n_polystyr. in vacuum')
plt.plot(wl0,m,label='n_polystyr. in water')
plt.legend()
plt.savefig(opj(figdir,'polystyrene_refractive_index.png'), dpi=300)

wl_water = wl0 / n_w
x_ = 2 * np.pi * radius_nm / wl_water

b_coef = concentration * xQeff.Qsca.isel(nr=range(-19, 0), ni=0).interp(x=x_) * np.pi * radius_mic ** 2
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
    b_coef_r = concentration * xQeff.Qsca.isel(nr=range(-19, 0), ni=0).interp(x=x_) * np.pi * radius ** 2
    b_coef_ = b_coef_r.interp(nr=m)

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

att_coef[sample].plot(x='wavelength', color='red', lw=3.5, label='measured', ax=axs)
# plt.legend()
for nr in b_coef.nr.values[0:-2:3]:
    axs.plot(b_coef.wavelength, b_coef.sel(nr=nr), color=cmap(norm(nr)), alpha=0.6, lw=1.5)

b_coef_beads.plot.line(x='wavelength', color='black', ls=':', lw=1.8, label='computed', ax=axs)
b_coef_int.plot.line(x='wavelength', color='black', ls='-', lw=2.3, label='computed with size ditrib.', ax=axs)

axs.set_title('')
axs.set_ylabel('$Scattering\ coef.\ (m^{-1})$')
axs.set_xlabel('$Wavelength\ (nm)$')
plt.legend()
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
cb.set_label('$n_r$', fontsize=22)
plt.suptitle(sample)
plt.savefig(opj(figdir,'scat_coef_beads_' + sample + suff + '.png'), dpi=300)


#----------------------------------------------
# effect of temperature or salinity
#----------------------------------------------

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen','orange','red'])
norm = mpl.colors.Normalize(vmin=0, vmax=40)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), sharey=True)

att_coef[sample].plot(x='wavelength', color='red', lw=3.5, label='measured', ax=axs)
# plt.legend()
for salinity in [0, 10, 20, 30, 40]:
    wl = wl0.data.astype(float)
    n_w = n_water_perso(wl, water_temp=20, water_sal=salinity)
    wl_water = wl0 / n_w

    n_p = n_polystyrene(wl0_mic)
    m = n_p / n_w
    arr = []
    for ir, radius in enumerate(radius_):
        wgt = ndp[ir]
        rad_nm = radius * 1000
        x_ = 2 * np.pi * rad_nm / wl_water
        b_coef_r = concentration * xQeff.Qsca.isel(nr=range(-19, 0), ni=0).interp(x=x_) * np.pi * radius ** 2
        b_coef_ = b_coef_r.interp(nr=m)


        arr.append(b_coef_.assign_coords(radius=radius) * wgt)

    b_coef_ = xr.concat(arr, dim='radius')
    b_coef_int = b_coef_.integrate('radius').compute()

    axs.plot(b_coef_int.wavelength, b_coef_int, color=cmap(norm(salinity)), alpha=0.6, lw=2)


axs.set_title('')
axs.set_ylabel('$Scattering\ coef.\ (m^{-1})$')
axs.set_xlabel('$Wavelength\ (nm)$')
plt.legend()
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top')
#cb.set_label('$Water\ temperature\ (\deg C)$', fontsize=22)
cb.set_label('$Water\ salinity\ (psu)$', fontsize=22)

plt.suptitle(sample)
plt.savefig(opj(figdir,'scat_coef_beads_' + sample + suff + '_water_salinity.png'), dpi=300)

# x_ = 2*np.pi *radius_nm /(att_coef.wavelength/ 1.334)
# b_coef = concentration * xQeff.Qsca.isel(nr=[-6,-5],ni=0).interp(x=x_) *np.pi*radius_mic**2
# b_coef.plot.line(x='wavelength')
