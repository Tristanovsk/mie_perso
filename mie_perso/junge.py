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
psd =psd.PSD


# -------------------------------------
# Generate mueller matrices for a series
# of size parameters x = np.pi * diameter / wavelength (unitless)
# -------------------------------------
m = 1.05 - 0.001j
#m = 1.20 - 0.000j
theta = np.concatenate([[0, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3], np.linspace(0.5, 179.5, 500),
                        [179.7, 179.8, 179.9, 179.95, 179.98, 179.99, 180]]) * np.pi / 180
theta = np.linspace(0, np.pi, 3600)
x = np.logspace(-1, 4, 501)
x = np.concatenate([np.logspace(-1, 2, 251), np.linspace(1e2 + 3, 3500, 501)])
x = np.concatenate([np.logspace(-1, 2, 251), np.logspace(np.log10(103), np.log10(1000), 251),
                    np.logspace(np.log10(1025), np.log10(3500), 125)])
gamma = 1.85
x = np.linspace(1e-2 ** (1 / gamma), 3500 ** (1 / gamma), 10001) ** gamma

x = np.linspace(1,40000,1001)
ofile = 'mie_perso/data/mueller_mie_' + str(m)[1:-1] + '_t3600_x1001.nc'
ofile = 'mie_perso/data/mueller_mie_' + str(m)[1:-1] + '_t7200_x10001.nc'
ofile = 'mie_perso/data/mueller_mie_' + str(m)[1:-1] + '_supres_xloglin.nc'

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

nMedium = 1.334
theta = mueller.theta
ang = theta * 180. / np.pi
Ntheta = len(mueller.theta)
rmin = .01
rmax = .5e2
slope = 4
wavelength=650
wavelength_medium = wavelength / nMedium
m = complex(mueller.nr, mueller.ni)
m_vacuum = m * nMedium

rmaxs = [20,50,100]
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(23, 8), sharex=True)
fig.subplots_adjust(bottom=0.115, top=0.88, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.15)
for imax,rmax in enumerate(rmaxs):
    axs[ imax].semilogy()
    axs[ imax].minorticks_on()
    axs[ imax].set_title('$r_{max} =$'+str(rmax)+'$\mu m$' )

    for imin,rmin in enumerate([1e-2,1e-1,1e0]):#[400, 500, 600, 700]:


        dpnm = mueller.x * wavelength_medium / np.pi
        dp = dpnm / 1000

        ndp = psd.power_law_junge(dp / 2, slope=-slope, rmin=rmin, rmax=rmax)
        # ndp = modif_power_law(dp / 2, slope=-slope, rmin=rmin, rmax=rmax)
        # convert to xarray
        ndp = dp.copy(data=ndp)
        # ndp = psd #[:-1]*np.diff(dp/2)
        S11, S12, S33, S34 = np.zeros(Ntheta), np.zeros(Ntheta), np.zeros(Ntheta), np.zeros(Ntheta)

        # aSDn = np.pi*((dp/2)**2)*ndp
        aSDn = ndp
        S11 = np.trapz(mueller.S11 * aSDn, dp, axis=0)

        norm = trapz(S11 * np.sin(theta), theta) / 2
        axs[ imax].plot(ang, S11 / norm, lw=1, label='$r_{min} = $'+str(rmin)+'$\mu m$')


    axs[imax].set_xlabel('$Scattering\ angle\ (deg)$')
axs[ 0].set_ylabel('S11')
# add outcomes from Mishchenko spher code
mat = pd.read_csv('./mie_perso/data/Mie_shchenko_ScatMat_junge_slope-4.00_rmin0.010_rmax20.0_nr1.050_ni-.0010_wl650.0.txt',
                  sep='\s+', skiprows=6, header=None)
axs[0].plot(mat.values[:, 0], mat.values[:, 3], 'k',label='NASA-GISS rmin 0.01')
mat = pd.read_csv('./mie_perso/data/Mie_shchenko_ScatMat_junge_slope-4.00_rmin0.010_rmax50.0_nr1.050_ni-.0010_wl650.0.txt',
                  sep='\s+', skiprows=6, header=None)
axs[1].plot(mat.values[:, 0], mat.values[:, 3], 'k',label='NASA-GISS rmin 0.01')
mat = pd.read_csv('./mie_perso/data/Mie_shchenko_ScatMat_junge_slope-4.00_rmin0.010_rmax100.0_nr1.050_ni-.0010_wl650.0.txt',
                  sep='\s+', skiprows=6, header=None)
axs[2].plot(mat.values[:, 0], mat.values[:, 3], 'k',label='NASA-GISS rmin 0.01')
for imax,rmax in enumerate(rmaxs):
    axs[ imax].legend()
plt.suptitle('Hydrosol, m='+str(m)[1:-1]+', Junge slope='+str(slope)+', wavelength = '+str(wavelength)+'nm')
plt.savefig('mie_perso/fig/junge_sensitivity_slope' + str(slope) + '_m' + str(m)[1:-1] + '_mw1.334_wl'+str(wavelength)+'nm.png', dpi=300)
for imax,rmax in enumerate(rmaxs):
    axs[ imax].loglog()
plt.savefig('mie_perso/fig/junge_sensitivity_loglog_slope' + str(slope) + '_m' + str(m)[1:-1] + '_mw1.334_wl'+str(wavelength)+'nm.png', dpi=300)


vsf_names = ['pf_Arizonadust', 'pf_C.autotrophica', 'pf_C.closterium',
             'pf_D.salina', 'pf_K.mikimotoi', 'pf_S.cf.costatum', 'pf_Beads_3µm',
             'pf_Beads_300nm', 'pf_Formazin(2ntu)', 'pf_Formazin(4ntu)']
vsf = pd.read_csv('./mie_perso/data/tabulated_phase_function_feb2015_full.txt', skiprows=11, na_values='NA')
mat = pd.read_csv('./mie_perso/data/Mie_shchenko_ScatMat_junge_slope-3.40_rmin0.010_nr1.050_ni-.0010_wl400.0.txt',
                  sep='\s+', skiprows=6, header=None)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)
fig.subplots_adjust(bottom=0.115, top=0.962, left=0.086, right=0.98,
                    hspace=0.01, wspace=0.01)
nMedium = 1.334
theta = mueller.theta
ang = theta * 180. / np.pi
Ntheta = len(mueller.theta)
rmin = .01
rmax = 150
slope = 4
for wavelength in [400, 500, 600, 700]:#:
    wavelength_medium = wavelength / nMedium
    m = complex(mueller.nr, mueller.ni)
    m_vacuum = m * nMedium

    dpnm = mueller.x * wavelength_medium / np.pi
    dp = dpnm / 1000

    ndp = psd.power_law_junge(dp / 2, slope=-slope, rmin=rmin, rmax=rmax)
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
    axs[0, 0].plot(ang, S11 / norm, lw=1, label='$wl_0=$'+str(wavelength)+'$wl_w=$'+str(wavelength_medium))
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
for vsf_name in np.array(vsf_names)[[0,3]]:
    axs[0, 0].plot(vsf['Angle(deg)'], 4 * np.pi * vsf[vsf_name], '--', label=vsf_name)
axs[0, 0].plot(mat.values[:, 0], mat.values[:, 3], 'r:')
axs[0, 1].plot(mat.values[:, 0], mat.values[:, 5] / mat.values[:, 3], 'r:')
axs[1, 0].plot(mat.values[:, 0], mat.values[:, 4] / mat.values[:, 3], 'r:')
axs[0, 0].legend()
plt.savefig('mie_perso/fig/junge_slope' + str(slope) + '_m' + str(m)[1:-1] + '_mw1.334_rmin' + str(rmin) +
            '_rmax' + str(rmax) + 'log.png', dpi=300)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)

rmin = 1
rmax = 100
dpnm = mueller.x * wavelength_medium / np.pi
dp = dpnm / 1000
for slope in [2.5, 3.0, 3.5, 4, 4.5]:

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.125, top=0.925, left=0.175, right=0.95, hspace=0.05)
    for rmin in [0.001, 0.01, 0.1, 1]:
        # refractive index
        m = 1.40 - 0.001j
        # diameter bins of PSD

        ndp = psd.power_law_junge(dp / 2, slope=-slope, rmin=rmin, rmax=rmax)
        junge = size_param(dp / 2, ndp)
        plt.plot(dp / 2, ndp, label=junge.to_annotation())
    plt.suptitle('Junge power law, slope=' + str(slope))
    plt.minorticks_on()
    plt.grid()
    plt.xlabel('$radius\ (\mu m)$')
    plt.ylabel(r'$dN(r)/dr\ (\mu m^{-1} \ cm^{-3})$')
    plt.loglog()
    plt.legend(fontsize=12)
    plt.savefig('mie_perso/fig/junge_slope' + str(slope) + '.png',
                dpi=300)
plt.show()

plt.figure()

for wl in [400, 500, 600, 700]:
    theta, SL, SR, SU = p.SF_SD_mp(m, wl, dpnm, ndp, nMedium=1.334, normalization='total')
    deg = theta * 180 / np.pi
    plt.plot(deg, SU, label=str(wl) + 'nm')
    # plt.plot(deg,SL,'r--')
    # plt.plot(deg,SR,'r:')
plt.suptitle(junge.to_annotation())
plt.semilogy()
plt.legend()
plt.show()
plt.savefig('mie_perso/fig/junge_slope' + str(slope) + '_m' + str(m)[1:-1] + '_mw1.334_rmin' + str(rmin) + 'log.png',
            dpi=300)

# -------------------------------------
# Plot scattering functions
# for a series of size parameters x (2pi r / lambda)
# -------------------------------------

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'black',
                                                     'orangered', "firebrick", "darkred",'yellowgreen','forestgreen'])

# cmap = plt.cm.bone_r
# norm = mpl.colors.Normalize(vmin=0, vmax=500)
mueller_ = mueller.isel(x=[1,10,20,50,100,250, 300, 350, 400, 500,600,700,850, 1000])

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



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10),sharex=True )
fig.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.92,
                    hspace=0.1, wspace=0.05)

for x_, m_ in list(mueller_.groupby('x')):
    axs[0, 0].plot(ang, m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
    axs[0, 1].plot(ang, m_.S12 / m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
    axs[1, 0].plot(ang, m_.S33 / m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
    axs[1, 1].plot(ang, m_.S34 / m_.S11, color=cmap(norm(x_)), alpha=0.5, lw=1)
axs[0, 0].semilogy()
for i in [0,1]:
    axs[i,1].yaxis.set_label_position("right")
    axs[i,1].yaxis.tick_right()
    for j in [0,1]:
        axs[i,j].minorticks_on()
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
