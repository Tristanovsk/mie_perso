
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
# Load mueller matrix LUT
# of size parameters x = np.pi * diameter / wavelength (unitless)
# -------------------------------------
ofile = 'data/mueller_mie_microplastic_v4.nc'
qfile = 'data/Qeff_mie_microplastic_v4.nc'
# lazy loading with dask, one chunk per refractive index (larger than memory LUT)
mueller = xr.open_dataset(ofile, chunks={'nr': 1})
xQeff = xr.open_dataset(qfile)

# forward peak on a fine angular grid (the 0.05 deg step of the LUT does not
# resolve the diffraction peak of the large particles), computed once
ffile = 'data/mueller_mie_microplastic_v4_forward.nc'
if not os.path.exists(ffile):
    theta_fwd = size_average.forward_grid()
    mueller_fwd = xr.concat([pmie.ScatMat_mp(complex(nr_, float(mueller.ni)), mueller.x.values, theta_fwd)
                             for nr_ in mueller.nr.values], 'nr')
    mueller_fwd.to_netcdf(ffile)
mueller_fwd = xr.open_dataset(ffile, chunks={'nr': 1})

nMedium = 1.334
wl = 800
wl_medium = wl / nMedium
radius_super =  np.logspace(-2, np.log10(1e5), 5001)
x = mueller.x
theta = mueller.theta
ang = theta * 180. / np.pi
Ntheta = len(mueller.theta)

radius_nm = x * wl_medium / (2 * np.pi)
radius_mic = 1e-3 * radius_nm
radius = radius_mic

# test size ditribution vs size parameter x
wavelengths_nm = np.arange(400,851,25)
rmed_list = [1., 10., 100., 1e3]
sigma = 1
#__________
# plot PSD
radius_super=np.logspace(-2,4,10000)
for i_rmed, rmed in enumerate(rmed_list):
    # compute size distribution
    ndp = psd.lognorm(radius_super, rn_med=rmed, sigma=sigma)* np.pi * radius_super
    plt.figure(figsize=(7, 4))
    handles = [mpl.patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                 lw=0, alpha=0)]
    ax = plt.subplot(111)
    ax.plot(radius_super, ndp,'k')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.semilogx()
    ax.set_xlabel('$Radius\ (\mu m)$')
    ax.minorticks_on()
    ax.legend(handles, [size_param(radius_super, ndp).to_annotation_r()], loc='best', fancybox=False,handlelength=0)
    plt.tight_layout()
    plt.savefig('./mie_perso/fig/PSD_rmed'+str(rmed)+'.png',dpi=300)

radius_super=np.logspace(-2,4,40000)
plt.figure(figsize=(8, 4))
handles = [mpl.patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
ax = plt.subplot(111)
colors=['darkred','red','khaki','lightgray']
zorders=[6,5,4,3]
for i_rmed, rmed in enumerate(rmed_list):
    # compute size distribution
    ndp = psd.lognorm(radius_super, rn_med=rmed, sigma=sigma)* np.pi * radius_super

    ax.plot(radius_super, ndp,lw=4,color=colors[i_rmed],zorder=zorders[i_rmed])

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.semilogx()
    ax.set_xlabel('$Radius\ (\mu m)$')
    ax.minorticks_on()
plt.tight_layout()
plt.savefig('./mie_perso/fig/all_PSD_detectability.png',dpi=300)

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', 'lightskyblue',
                                                     'orangered', "firebrick", "darkred",
                                                     'yellowgreen', 'forestgreen'])

fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharex=True, sharey='row')
fig.subplots_adjust(bottom=0.115, top=0.98, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.05)
norm = mpl.colors.LogNorm(vmin=1.05, vmax=1.2)
norm = mpl.colors.Normalize(vmin=1.01, vmax=1.21)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

lw = 1.2
alpha = 0.5
suff = ""
# suff='_full'
first = True

# compute mean volume of the particles in micron**-3


geom_=[]
radius =radius_super
for i_rmed, rmed in enumerate(rmed_list):
    # compute size distribution
    ndp = psd.lognorm(radius, rn_med=rmed, sigma=sigma)

    # compute mean volume of the particles in micron**-3
    ndp_norm = trapezoid(ndp, radius)
    vol_mean = trapezoid(ndp * 4 / 3 * np.pi * radius ** 3, radius) / ndp_norm
    geom_.append([rmed,vol_mean])
geom = pd.DataFrame(geom_,columns=['r_med','vol_mean']).set_index(['r_med']).to_xarray().assign_attrs({'description':'Mean volume of a particle (micron**3)'})

# integrate mueller matrix and cross sections over the size distributions
# for all the wavelengths and r_med in a single pass over the LUT
# (radius and wavelength in micron, then indexed by the wavelength in vacuum in nm)
weights = size_average.psd_family_weights(
    mueller.x, wavelengths_nm * 1e-3, lambda r, rmed: psd.lognorm(r, rn_med=rmed, sigma=sigma),
    rmed_list, 'r_med', n_medium=nMedium).assign_coords(wavelength=wavelengths_nm)

# scattering matrix and cross sections in micron**2
# merged with the forward peak: 1/2 int S11 sin(theta) dtheta = Csca
xscat_mat = size_average.merge_forward(size_average.integrate_scatmat(mueller, weights),
                                       size_average.integrate_scatmat(mueller_fwd, weights)).compute()
theta = xscat_mat.theta
ang = theta * 180. / np.pi
xCrossSection = size_average.integrate_cross_section(xQeff.isel(ni=0), weights, variables=('Qsca', 'Qext')).compute()
# TODO save into netcdf with good file naming

# -------------------------------------
# Integrate over refractive index distribution
#
# -------------------------------------

nr = xscat_mat.nr

# nr = np.linspace(1.05,1.2,100)
# nr=xr.DataArray(nr,coords=dict(nr=nr))
# mueller_int = mueller_int.interp(nr=nr)
nr_=1.12
nr_dist = psd.norm(nr, nr_, 0.05)
(xscat_mat * nr_dist).integrate('nr')

ax = plt.subplot(111)
ax.plot(nr, nr_dist,'k')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_ticks([])
ax.set_xlabel('$Refractive\ index$')

#plt.savefig('./mie_perso/fig/refrac_index_distrib.png',dpi=300)
trapezoid(nr_dist, nr)
sigma = 0.5
lw = 1.2
alpha = 0.5

xscat_mat_ = xscat_mat.sel(wavelength=800,r_med=1000)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.2)
for nr_ in [1.05, 1.09, 1.145]:
    nr_dist = psd.norm(nr, nr_, 0.05)
    mueller_super = (xscat_mat_ * nr_dist).integrate('nr')
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



cmap = plt.cm.Spectral_r
norm = mpl.colors.Normalize(vmin=400, vmax=800)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharex=True, sharey='row')
fig.subplots_adjust(bottom=0.115, top=0.98, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.05)

nr_=1.20
first=True
for wavelength in xscat_mat.wavelength.data:
    for i_rmed, rmed in enumerate(xscat_mat.r_med.data):
        xscat_mat_ = xscat_mat.sel(r_med=rmed,wavelength=wavelength)
        vol_mean = geom.vol_mean.sel(r_med=rmed).data
        nr_dist = psd.norm(nr, nr_, 0.05)
        nr_dist_norm = trapezoid(nr_dist,nr)

        mueller_super = (xscat_mat_ * nr_dist).integrate('nr')/nr_dist_norm

        Cross = xCrossSection.sel(r_med=rmed,wavelength=wavelength)
        Csca = (Cross.Csca* nr_dist).integrate('nr')/nr_dist_norm
        Cext = (Cross.Cext* nr_dist).integrate('nr')/nr_dist_norm

        S11 = mueller_super.S11
        Cscat = trapezoid(S11 * np.sin(theta), theta) / 2
        print(Cext.values/vol_mean,Csca.values/vol_mean,Cscat/vol_mean)

        S12 = mueller_super.S12 / S11
        S33 = mueller_super.S33 / S11
        axs[0, i_rmed].plot(ang, S11 / Csca, color=cmap(norm(wavelength)), lw=lw, alpha=alpha)
        axs[1, i_rmed].plot(ang, S12, color=cmap(norm(wavelength)), lw=lw, alpha=alpha)
        axs[2, i_rmed].plot(ang, S33, color=cmap(norm(wavelength)), lw=lw, alpha=alpha)
        if first:
            axs[0, i_rmed].set_title('$Lognorm,\ r_m = ' + str(rmed) + ',\ sig=$' + str(sigma))

    first = False
axs[1,0].set_ylim([-1.05, 1.05])
axs[2,0].set_ylim([-1.05, 1.05])
axs[0,0].semilogy()
for i in range(4):
        axs[-1, i].set_xlabel('$Scattering\ angle\ (deg)$')
axs[0, 0].set_ylim([5e-3, 4e4])
# axs[0, 0].legend()
axs[0, 0].set_ylabel('$S_{11}$')
axs[1, 0].set_ylabel('$S_{12}$')
axs[2, 0].set_ylabel('$S_{33}$')
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.05, location='top', format=mpl.ticker.ScalarFormatter())
cb.set_label('$Wavelength\ (nm)$', fontsize=22)

plt.savefig('./mie_perso/fig/scat_mat_psd_lognorm_osoaa'+str(nr_)+'.png', dpi=300)

# ----------------------------------------
# Generate OSOAA scat mat file
nr_=1.15
first=True
for wavelength in xscat_mat.wavelength.data:
    for i_rmed, rmed in enumerate(xscat_mat.r_med.data):

        ofile = './mie_perso/lut/scat_mat_psd_lognorm_osoaa_nr'+str(nr_)+'_rmed'+str(rmed)+'_wl'+str(wavelength)+'.csv'

        xscat_mat_ = xscat_mat.sel(r_med=rmed,wavelength=wavelength)
        vol_mean = geom.vol_mean.sel(r_med=rmed).data
        nr_dist = psd.norm(nr, nr_, 0.05)
        nr_dist_norm = trapezoid(nr_dist,nr)

        mueller_super = (xscat_mat_ * nr_dist).integrate('nr')/nr_dist_norm

        Cross = xCrossSection.sel(r_med=rmed,wavelength=wavelength)
        Csca = (Cross.Csca* nr_dist).integrate('nr')/nr_dist_norm
        Cext = (Cross.Cext* nr_dist).integrate('nr')/nr_dist_norm

        S11 = mueller_super.S11
        Cscat = trapezoid(S11 * np.sin(theta), theta) / 2
        print(Cext.values/vol_mean,Csca.values/vol_mean,Cscat/vol_mean)

        S12 = mueller_super.S12 / S11
        S33 = mueller_super.S33 / S11
        S11 = S11 / Cscat
        # set S22 to one
        S22=S11*0+1
        mat = xr.Dataset({'s11':S11,'s12':S12,'s22':S22,'s33':S33})
        mat = mat.assign_coords(theta=np.degrees(mat.theta))
        df_mat = mat.reset_coords(drop=True).to_pandas()
        Nang = len(mueller_super.theta)

        Cext_ = "{:12.6e}".format(Cext.values)
        Csca_ = "{:12.6e}".format(Csca.values)
        V_mean_ = "{:12.6e}".format(vol_mean)
        # check normalization:
        theta_rad = np.deg2rad(df_mat.index)
        print(trapezoid(df_mat.s11 * np.sin(theta_rad), theta_rad))


        header = 'Cext : '+Cext_+'\nCsca : '+Csca_+'\nvol_mean : '+V_mean_+\
                 '\nNb angles : '+str(Nang)+'\n'

        with open(ofile, 'w') as fp:
            fp.write(header)
        df_mat.to_csv(ofile,sep=' ', header=True, mode='a')




