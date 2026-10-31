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
# Generate mueller matrices for a series
# of size parameters x = np.pi * diameter / wavelength (unitless)
# -------------------------------------
# list of refractive index in medium
mi = 0.
m_list = [complex(1.05, 0), complex(1.07, 0), complex(1.09, 0), complex(1.11, 0),
          complex(1.13, 0), complex(1.15, 0), complex(1.17, 0), complex(1.20, 0)]

theta = np.linspace(0, np.pi, 3600)
x = np.linspace(1, 40000, 1001)
x = np.logspace(-1, np.log10(40000), 501)
# x = np.logspace(-1, np.log10(400), 21)

# test x for given wl and psd:
nMedium = 1.334
wl = 600
wl_medium = wl / nMedium
radius_nm = x * wl_medium / (2 * np.pi)
radius_mic = 1e-3 * radius_nm
radius_mm = 1e-3 * radius_mic

ofile = 'data/mueller_mie_microplastic_v4.nc'
qfile = 'data/Qeff_mie_microplastic_v4.nc'


if os.path.exists(ofile):
    mueller = xr.open_dataset(ofile, chunks={'nr': 1})
    xQeff=xr.open_dataset(qfile)
else:
    m_ = []
    mi = 0.
    mr_list = np.arange(1.005, 1.22, 0.005)
    # generate scattering matrix
    for mr in mr_list:
        m = complex(mr, mi)
        res = pmie.ScatMat_mp(m, x, theta)
        ofile_ = 'data/mueller_mie_microplastic' + format(m, '1.3f') + '_v4.nc'
        res.to_netcdf(ofile_)
        m_.append(res)
    mueller = xr.concat(m_, "nr")
    mueller.to_netcdf(ofile)

    # generate Qeff
    # use of fake wavelength and diameter based on x
    Qeff = []
    wl = 500

    for x_ in x:
        p=[]
        for mr in mr_list:
            m=complex(mr, mi)
            diameter = wl *  x_ /np.pi
            p.append([x_,mr, mi])
        with multiprocessing.Pool(processes=pmie.processes) as pool:
            res = pool.map(pmie.Qeff_compute, p)

        Qeff.append(res)


    xQeff=pd.DataFrame(np.concatenate(Qeff),
                       columns=['x','mr','mi','Qext', 'Qsca', 'Qabs', 'g', 'Qpr', 'Qback', 'Qratio']
                       ).set_index(['x','mr','mi']).to_xarray()
    qfile = 'data/Qeff_mie_microplastic_v4.nc'

    xQeff.to_netcdf(qfile)


# m_=[]
# for m in m_list[:2]:
#     ofile_ = 'data/mueller_mie_microplastic' + format(m,'1.3f') + '.nc'
#     m_.append(xr.open_dataset(ofile_))
# mueller = xr.concat(m_, "nr")
wl = 800
mueller = mueller.dropna('x')
mueller = mueller.chunk(chunks={'nr': 1})
x = mueller.x

radius_nm = x * wl_medium / (2 * np.pi)
radius_mic = 1e-3 * radius_nm
radius_mm = 1e-3 * radius_mic
radius = radius_mm
plt.figure()
for gamma in [0.2, 0.3, 0.4]:
    psd_ = psd.microplastic_AF2021(radius, gamma=gamma)
    plt.plot(2 * radius, psd_, '.', label='$\gamma = $' + str(gamma))
plt.legend()
plt.semilogx()
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
plt.figure()
first = True
for wavelength in [400, 500, 600, 700, 800]:
    wavelength_medium = wavelength / nMedium
    radius_nm = mueller.x * wavelength_medium / np.pi / 2
    radius_mic = 1e-3 * radius_nm
    radius_mm = 1e-3 * radius_mic

    for i, radius in enumerate([radius_mic, radius_mm]):
        ndp = psd.microplastic_AF2021(radius, gamma=gamma)

        plt.plot(radius.x, ndp, label=str(i))
    plt.semilogx()
    plt.legend()

size_class = 'large'
size_class='small'
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', 'lightskyblue',
                                                     'orangered', "firebrick", "darkred",
                                                     'yellowgreen', 'forestgreen'])

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey='row')
fig.subplots_adjust(bottom=0.115, top=0.98, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.05)
norm = mpl.colors.LogNorm(vmin=1.05, vmax=1.2)
norm = mpl.colors.Normalize(vmin=1.01, vmax=1.21)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
sigma = 0.5
lw = 1.2
alpha = 0.5

first = True
# integrate over the size distributions for all wavelengths and gammas in one pass over the LUT
wavelengths = [400, 500, 600, 700, 800]
gammas = [ 1., 10. ,100.]
radius_unit = 1e-3 if size_class == 'large' else 1.  # psd radius in mm or micron
weights = size_average.psd_family_weights(
    mueller.x, np.array(wavelengths) * 1e-3, lambda r, g: psd.lognorm(r * radius_unit, rn_med=g, sigma=sigma),
    gammas, 'gamma', n_medium=nMedium).assign_coords(wavelength=wavelengths)
mueller_psd = size_average.integrate_scatmat(mueller.isel(nr=[9, 19, 29, 39]), weights).compute()
Csca_psd = size_average.integrate_cross_section(xQeff.isel(nr=[9, 19, 29, 39], ni=0), weights, variables=('Qsca',)).Csca.compute()
for wavelength in wavelengths:
    wavelength_medium = wavelength / nMedium
    radius_nm = mueller.x * wavelength_medium / np.pi / 2
    radius_mic = 1e-3 * radius_nm
    radius_mm = 1e-3 * radius_mic

    if size_class == "large":
        radius = radius_mm
    else:
        radius = radius_mic
    # for igamma, gamma in enumerate([0.2, 0.3, 0.4]):
    #     ndp = psd.microplastic_AF2021(radius, gamma=gamma)
    for igamma, gamma in enumerate(gammas):

        mueller_int = mueller_psd.sel(wavelength=wavelength, gamma=gamma)
        Csca_int = Csca_psd.sel(wavelength=wavelength, gamma=gamma)

        for nr in mueller_int.nr.data:
            # mueller_ = mueller #.rolling(theta=5, center=True).median()
            _mueller = mueller_int.sel(nr=nr)
            S11 = _mueller.S11  # micron**2
            S12 = _mueller.S12 / _mueller.S11
            S33 = _mueller.S33 / _mueller.S11
            S34 = _mueller.S34 / _mueller.S11

            Csca_fromS11 = trapezoid(S11 * np.sin(theta), theta) / 2
            Csca = Csca_int.sel(nr=nr)
            #Csca= Csca_fromS11
            if first:
                axs[0, igamma].plot(ang, S11 / Csca, color=cmap(norm(nr)), lw=lw, alpha=alpha,
                                    label='$n_r= $' + str(nr))
            else:
                axs[0, igamma].plot(ang, S11 / Csca, color=cmap(norm(nr)), lw=lw, alpha=alpha)

            axs[1, igamma].plot(ang, S12, color=cmap(norm(nr)), lw=lw, alpha=alpha, label='$n_r= $' + str(nr))
            axs[2, igamma].plot(ang, S33, color=cmap(norm(nr)), lw=lw, alpha=alpha, label='$n_r= $' + str(nr))
            # axs[3,igamma].plot(ang, S34, lw=1,alpha=0.4, label='$n_r= $' + str(nr) )
        if first:
            axs[0, igamma].set_title('$Lognorm,\ r_m = ' + str(gamma) + ',\ sig=$' + str(sigma))

    first = False

    for i in range(3):
        axs[0, i].semilogy()

        axs[1, i].set_ylim([-1.05, 1.05])
        axs[2, i].set_ylim([-1.05, 1.05])
        axs[-1, i].set_xlabel('$Scattering\ angle\ (deg)$')
# axs[0, 0].legend()
axs[0, 0].set_ylabel('$S_{11}$')
axs[1, 0].set_ylabel('$S_{12}$')
axs[2, 0].set_ylabel('$S_{33}$')
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top', format=mpl.ticker.ScalarFormatter())
cb.set_label('$n_r$', fontsize=22)

plt.savefig('./mie_perso/fig/mueller_mat_psd_lognorm_' + size_class + '_v2.png', dpi=300)

# ---------------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharex=True, sharey='row')
fig.subplots_adjust(bottom=0.115, top=0.98, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.05)
norm = mpl.colors.LogNorm(vmin=1.05, vmax=1.2)
norm = mpl.colors.Normalize(vmin=1.01, vmax=1.21)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
sigma = 0.5
lw = 1.2
alpha = 0.5
suff=""
suff='_full'
first = True
# integrate over the size distributions for all wavelengths and rmed in one pass over the LUT
wavelengths = [500]
rmeds = [1.,10.,100.,1e3]
nr_sel = slice(None) if suff == '_full' else [9, 19, 29, 39]
weights = size_average.psd_family_weights(
    mueller.x, np.array(wavelengths) * 1e-3, lambda r, rmed: psd.lognorm(r, rn_med=rmed, sigma=sigma),
    rmeds, 'r_med', n_medium=nMedium).assign_coords(wavelength=wavelengths)
mueller_psd = size_average.integrate_scatmat(mueller.isel(nr=nr_sel), weights).compute()
Csca_psd = size_average.integrate_cross_section(xQeff.isel(nr=nr_sel, ni=0), weights, variables=('Qsca',)).Csca.compute()
for wavelength in wavelengths:#400, 500, 600, 700, 800]:
    wavelength_medium = wavelength / nMedium
    radius_nm = mueller.x * wavelength_medium / np.pi / 2
    radius_mic = 1e-3 * radius_nm
    radius = radius_mic


    # for i_rmed, rmed in enumerate([0.2, 0.3, 0.4]):
    #     ndp = psd.microplastic_AF2021(radius, rmed=rmed)
    for i_rmed, rmed in enumerate(rmeds):

        mueller_int = mueller_psd.sel(wavelength=wavelength, r_med=rmed)
        Csca_int = Csca_psd.sel(wavelength=wavelength, r_med=rmed)

        for nr in mueller_int.nr.data:
            # mueller_ = mueller #.rolling(theta=5, center=True).median()
            _mueller = mueller_int.sel(nr=nr)
            S11 = _mueller.S11  # micron**2
            S12 = _mueller.S12 / _mueller.S11
            S33 = _mueller.S33 / _mueller.S11
            S34 = _mueller.S34 / _mueller.S11

            Csca_fromS11 = trapezoid(S11 * np.sin(theta), theta) / 2
            Csca = Csca_int.sel(nr=nr)
            #Csca= Csca_fromS11
            if first:
                axs[0, i_rmed].plot(ang, S11 / Csca, color=cmap(norm(nr)), lw=lw, alpha=alpha,
                                    label='$n_r= $' + str(nr))
            else:
                axs[0, i_rmed].plot(ang, S11 / Csca, color=cmap(norm(nr)), lw=lw, alpha=alpha)

            axs[1, i_rmed].plot(ang, S12, color=cmap(norm(nr)), lw=lw, alpha=alpha, label='$n_r= $' + str(nr))
            axs[2, i_rmed].plot(ang, S33, color=cmap(norm(nr)), lw=lw, alpha=alpha, label='$n_r= $' + str(nr))
            # axs[3,i_rmed].plot(ang, S34, lw=1,alpha=0.4, label='$n_r= $' + str(nr) )
        if first:
            axs[0, i_rmed].set_title('$Lognorm,\ r_m = ' + str(rmed) + ',\ sig=$' + str(sigma))

    first = False

    for i in range(4):
        axs[0, i].semilogy()

        axs[1, i].set_ylim([-1.05, 1.05])
        axs[2, i].set_ylim([-1.05, 1.05])
        axs[-1, i].set_xlabel('$Scattering\ angle\ (deg)$')
axs[0, 0].minorticks_on()
axs[0, 0].set_ylim([5e-6, 4e4])
# axs[0, 0].legend()
axs[0, 0].set_ylabel('$F_{11}$')
axs[1, 0].set_ylabel('$F_{12}/F_{11}$')
axs[2, 0].set_ylabel('$F_{33}/F_{11}$')
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top', format=mpl.ticker.ScalarFormatter())
cb.set_label('$m_r$', fontsize=26)

plt.savefig('./mie_perso/fig/mueller_mat_psd_lognorm_osoaa'+suff+'_500nm_sig' + str(sigma)+'.png', dpi=300)

# plt.savefig('./fig/mueller_mat_psd_'+size_class+'.png',dpi=300)

# test Cscat computation

radius = radius_mm
# microplastic psd (radius in mm) at the last wavelength, all nr in one pass over the LUT
weights = size_average.psd_family_weights(
    mueller.x, np.array([wavelength]) * 1e-3, lambda r, g: psd.microplastic_AF2021(r * 1e-3, gamma=g),
    [0.2, 0.3, 0.4], 'gamma', n_medium=nMedium)
mueller_psd = size_average.integrate_scatmat(mueller, weights).compute()
Cscat = (mueller_psd.S11 * np.sin(theta)).integrate('theta') / 2
Csca_psd = size_average.integrate_cross_section(xQeff.isel(ni=0), weights, variables=('Qsca',)).Csca
print('Csca from F11 and from Qsca (micron**2):', Cscat.values, Csca_psd.values)

# -------------------------------------
# Plot integrate over refractive index distribution
#
# -------------------------------------
radius = radius_mic
weights = size_average.psd_weights(mueller.x, wavelength * 1e-3,
                                   lambda r: psd.lognorm(r, rn_med=4, sigma=sigma), n_medium=nMedium)
mueller_int = size_average.integrate_scatmat(mueller, weights).isel(wavelength=0).compute()
nr = mueller_int.nr  # .values

# nr = np.linspace(1.05,1.2,100)
# nr=xr.DataArray(nr,coords=dict(nr=nr))
# mueller_int = mueller_int.interp(nr=nr)
nr_dist = psd.norm(nr, 1.05, 0.05)
plt.figure()
plt.plot(nr, nr_dist)
trapezoid(nr_dist, nr)
sigma = 0.5
lw = 1.2
alpha = 0.5
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.2)
for nr_ in [1.05, 1.09, 1.145]:
    nr_dist = psd.norm(nr, nr_, 0.05)
    mueller_super = (mueller_int * nr_dist).integrate('nr')
    S11 = mueller_super.S11
    Cscat = trapezoid(S11 * np.sin(theta), theta) / 2
    S12 = mueller_super.S12 / S11
    S33 = mueller_super.S33 / S11
    axs[0].plot(ang, S11 / Cscat, color=cmap(norm(nr_)), lw=lw, alpha=alpha)
    axs[1].plot(ang, S12, color=cmap(norm(nr_)), lw=lw, alpha=alpha)
    axs[2].plot(ang, S33, color=cmap(norm(nr_)), lw=lw, alpha=alpha)
axs[1].set_ylim([-1.05, 1.05])
axs[2].set_ylim([-1.05, 1.05])
axs[0].semilogy()
for i in range(3):
    axs[i].set_xlabel('$Scattering\ angle\ (deg)$')

first = True
wavelengths = [500, 600]
radius_unit = 1e-3 if size_class == 'large' else 1.  # psd radius in mm or micron
weights = size_average.psd_family_weights(
    mueller.x, np.array(wavelengths) * 1e-3, lambda r, g: psd.lognorm(r * radius_unit, rn_med=g, sigma=sigma),
    [0.2, 1., 4], 'gamma', n_medium=nMedium).assign_coords(wavelength=wavelengths)
mueller_psd = size_average.integrate_scatmat(mueller, weights).compute()
S11_norm = (mueller_psd.S11 * np.sin(theta)).integrate('theta') / 2

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'black',
                                                     'orangered', "firebrick", "darkred", 'yellowgreen', 'forestgreen'])

# cmap = plt.cm.bone_r
# norm = mpl.colors.Normalize(vmin=0, vmax=500)
# one refractive index (nr=2) for a few size parameters
mueller_ = mueller.isel(x=[1, 10, 20, 50, 100, 150, 160, 170, 250], nr=2).compute()

norm = mpl.colors.LogNorm(vmin=mueller_.x.min(), vmax=mueller_.x.max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.985, left=0.1, right=0.98,
                    hspace=0.01, wspace=0.025)

for x_ in mueller_.x.values:
    m_ = mueller_.sel(x=x_)
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
fig.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.92,
                    hspace=0.1, wspace=0.05)

for x_ in mueller_.x.values:
    m_ = mueller_.sel(x=x_)
    axs[0, 0].plot(ang, m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
    axs[0, 1].plot(ang, m_.S12 / m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
    axs[1, 0].plot(ang, m_.S33 / m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
    axs[1, 1].plot(ang, m_.S34 / m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
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
axs[1, 0].set_xlabel('$Scattering\ angle\ (deg)$')
axs[1, 1].set_xlabel('$Scattering\ angle\ (deg)$')

axs[1, 0].set_ylim([-1.05, 1.05])
axs[1, 1].set_ylim([-1.05, 1.05])
axs[0, 1].set_ylim([-1.05, 1.05])
