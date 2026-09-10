import os

import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.integrate import trapz  # import a single function for integration using trapezoidal rule

import matplotlib.pyplot as plt
import matplotlib as mpl

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 22 })
from lmfit import models, report_fit


def psd(mu1=-2.813, sig1=0.4, ):
    '''

    :param amps: amps volume proportion of each mode (ex: [0.1,0.6,0.4], sum(amps)=1)
    :param mu1:
    :param sig1:
    :param mu2:
    :param sig2:
    :param mu3:
    :param sig3:
    :return:
    '''

    gauss1 = models.GaussianModel(prefix='g1_')
    pars = gauss1.make_params()
    #pars['g1_center'].set(min=np.log(0.001), max=np.log(0.5))
    pars['g1_center'].set(value=mu1)
    pars['g1_sigma'].set(value=sig1, min=0.1, max=.6)


    mod = gauss1
    return mod, pars


def amplitude_lognorm(mu, sig, height=1):
    return height * max(1e-19, (sig * np.sqrt(2 * np.pi))) / np.exp(sig ** 2 / 2 - mu)

def psd_log(mu1=-0.197, sig1=0.4):
    '''

    :param amps: amps volume proportion of each mode (ex: [0.1,0.6,0.4], sum(amps)=1)
    :param mu1:
    :param sig1:
    :param mu2:
    :param sig2:
    :param mu3:
    :param sig3:
    :return:
    '''

    gauss1 = models.LognormalModel(prefix='g1_')
    pars = gauss1.make_params()
    pars['g1_center'].set(min=-20, max=10)
    pars['g1_center'].set(value=mu1)
    pars['g1_sigma'].set(value=sig1, min=0.1, max=.6)

    mod = gauss1
    return mod, pars

def power_law_junge(r,slope=-3.5,rmin=0.03,rmax=100):
    psd=np.array(r**slope)
    psd[r < rmin]=0.
    psd[r > rmax]=0.
    return psd / np.trapz(psd,r)

def modif_power_law(r,slope=-3.5,rmin=0.03,rmax=100):
    psd=np.array((r/rmin)**slope)
    psd[r < rmin]=1
    psd[r > rmax]=0.
    return psd / np.trapz(psd,r)

def rmod2rmed(rmod, sig):
    return np.exp(np.log(rmod)+sig**2)

def rmed2rmod(rmed, sig):
    return np.exp(np.log(rmed)-sig**2)

def muv2mun(muv,sig):
    return muv-3*sig**2

def mun2muv(mun,sig):
    return mun+3*sig**2

def rvmod2rnmed(rv_mod, sig):
    return rv_mod*np.exp(-2* sig ** 2)

def rnmed2rvmed(rn_med, sig):
    return np.exp(np.log(rn_med)+3*sig**2)

def arr_format(arr, fmt="{:0.1f}"):
    return [fmt.format(x) for x in arr]

# --------------------------
# nonlinear fitting
r = np.logspace(-2, np.log10(5), 1000)
#r = np.logspace(-6, 2, 5000)
logr = np.log(r)

rv_med = 0.25
sig = 0.4
legs = ['$PSD_1$', '$PSD_2$', '$PSD_3$']
ls = ['-','--',':']

rv_mod = rmed2rmod(rv_med, sig)
rn_med = rvmod2rnmed(rv_mod, sig)
rn_mod = rmed2rmod(rn_med, sig)
muv=np.log(rv_med)


#----------------
# plot dN/dr and dV/dlogr
#----------------
junge_slopes=[-3.,-4.5]
junge_str=arr_format(junge_slopes)
idx=500
fig, ax = plt.subplots(1, 1, figsize=(11,7))#,sharex=True)
#fig.subplots_adjust(bottom=0.075, top=0.94, left=0.175, right=0.95,hspace=0.05)
#ax = axs[0]
ax.minorticks_on()
sig = 0.45
rn_meds = [0.5,1,2]
for i, rn_med in enumerate(rn_meds):

    mun=np.log(rn_med)
    mod, pars = psd_log( mu1=mun, sig1=sig)
    n_r = mod.eval(pars, x=r)
    #n_r = y / (4 / 3 * np.pi * r ** 3)
    ax.plot(r, n_r,   lw=2, label='mun='+str(mun),zorder=6)
    if i == 1:
        norm_log=n_r[idx]
    print(np.trapz(n_r,r))
pow_coarse = power_law_junge(r,slope=junge_slopes[0])
pow_fine = power_law_junge(r,slope=junge_slopes[1])
#
# pow_coarse = modif_power_law(r,slope=junge_slopes[0])
# pow_fine = modif_power_law(r,slope=junge_slopes[1])
# ax.plot(r,pow_coarse,  'r', lw=2,label='pow '+junge_str[0])
# ax.plot(r, pow_fine,  'g', lw=2,label='pow '+junge_str[1])
ax.set_ylabel(r'$dN(r)/dr\ (mm^{-1} \ m^{-3})$')
def r2x(r,wl=670e-6):
    return 2*np.pi*r /wl
def x2r(x,wl=670e-6):
    return x*wl / (2*np.pi)
secax = ax.secondary_xaxis('top', functions=(r2x,x2r))
secax.set_xlabel('x-parameter')
#ax.set_xlabel(r'$Radius\ (\mu m)$')
ax.legend()
ax.set_xlabel(r'$Volume-equivalent\ radius\ (mm)$')
plt.tight_layout()
ax.loglog()


ax.set_ylim([1e-9,1e2])

ax = axs[1]
ax.tick_params(axis='x', which='major', length=7,width=1.2)
ax.tick_params(axis='x', which='minor', length=4)
#ax2 = ax.twinx()
for i, type in enumerate(types):
    # mod, pars = psd(type)
    # y = mod.eval(pars, x=logr)
    # ax.plot(r, y, 'k',ls=ls[i], lw=2,label=legs[i])
    mod, pars = psd(type, mu1=muv[0], mu2=muv[1], mu3=muv[2],sig1=sig[0], sig2=sig[1],  sig3=sig[2])
    v_r = mod.eval(pars, x=logr)
    ax.plot(r, v_r, 'k', ls=ls[i], lw=2, label=legs[i],zorder=6)
# add power law for volume and variable logr (4/3*pi*r**3*n(r)/r)
pow_coarse_v=4/3*np.pi*r**2 * pow_coarse
pow_coarse_v=pow_coarse_v/trapz(pow_coarse_v,r)
pow_fine_v=4/3*np.pi*r**2 * pow_fine
pow_fine_v=pow_fine_v/trapz(pow_fine_v,r)
ax.plot(r, pow_coarse_v,  'r', lw=2,label='pow '+junge_str[0])
ax.plot(r, pow_fine_v,  'g', lw=2,label='pow '+junge_str[1])

#ax.vlines(rv_med,0,1,colors='r')

ax.set_ylabel(r'$v(r)\ (\mu m^3\ cm^{-3})$')
ax.semilogx()

ax.set_xlabel(r'$Radius\ (\mu m)$')
ax.set_xlim([0.03,100])
ax.set_ylim([0.0,0.8])

ax.legend()
plt.show()

plt.savefig('fig/PSD_dN_dr_dV_dlogr_SPM_power_law_'+lut+'.png',dpi=300)
#




#----------------
#plot dN/dr
#----------------

fig, axs = plt.subplots(1, 1, figsize=(7,6))
fig.subplots_adjust(bottom=0.15, top=0.97, left=0.175, right=0.95)
ax = axs
for i, type in enumerate(types):
    mod, pars = psd_log(type, mu1=mun[0], mu2=mun[1],  mu3=mun[2],sig1=sig[0], sig2=sig[1],  sig3=sig[2])
    n_r = mod.eval(pars, x=r)
    #n_r = y / (4 / 3 * np.pi * r ** 3)
    ax.plot(r, n_r/np.trapz(n_r,r),  'k',ls=ls[i], lw=2, label=legs[i])
    print(np.trapz(n_r,r))
norm_3 = np.trapz(r**-3,r)
norm_5 = np.trapz(r**-5,r)

ax.plot(r, r**-3/norm_3,  'r', lw=2)
ax.plot(r, r**-5/norm_5 ,  'g', lw=2)

ax.loglog()
ax.set_xlim([0.01,100])
ax.set_ylabel(r'$dN(r)/dr\ (\mu m^{-1} \ cm^{-3})$')
#ax.set_xlabel(r'$Radius\ (\mu m)$')
ax.legend()
plt.show()
plt.savefig('fig/PSD_dN_dr_SPM_number_'+lut+'.png',dpi=300)

#----------------
# plot dV/dlogr
#----------------
fig, axs = plt.subplots(1, 1, figsize=(7.7,5.85))
fig.subplots_adjust(bottom=0.15, top=0.97, left=0.15, right=0.95)
ax = axs
ax.tick_params(axis='x', which='major', length=7,width=1.2)
ax.tick_params(axis='x', which='minor', length=4)
#ax2 = ax.twinx()
for i, type in enumerate(types):
    # mod, pars = psd(type)
    # y = mod.eval(pars, x=logr)
    # ax.plot(r, y, 'k',ls=ls[i], lw=2,label=legs[i])
    mod, pars = psd(type, mu1=muv[0], mu2=muv[1], mu3=muv[2],sig1=sig[0], sig2=sig[1],  sig3=sig[2])
    v_r = mod.eval(pars, x=logr)
    ax.plot(r, v_r, 'k', ls=ls[i], lw=2, label=legs[i])

#     # number distribution
#     mod, pars = psd(type,mu1=mun[0], mu2=mun[1],  mu3=mun[2])
#     n_r = mod.eval(pars, x=logr)
#     #ax2.plot(r[:-1], v_r[:-1]*np.diff(logr), 'r', ls=ls[i], lw=2, label=legs[i])
#     ax2.plot(r, n_r, 'g',ls=ls[i], lw=2,label=legs[i])
#
#
# ax2.tick_params(axis='y', labelcolor='r')
# ax2.set_ylabel(r'$n(r)\ (cm^{-3})$',color='r')

ax.set_ylabel(r'$v(r)\ (\mu m^3\ cm^{-3})$')
ax.semilogx()

ax.set_xlabel(r'$Radius\ (\mu m)$')
ax.set_xlim([0.1,100])
ax.legend()
plt.savefig('fig/PSD_dV_dlogr_SPM_'+lut+'.png',dpi=300)



# #----------------
# #plot dN/dlogr
# #----------------
#
#
# types = [[0.99, 0.006, 0.004], [0.45, 0.5, 0.05], [0.9, 0.05, 0.05]]
#
# fig, axs = plt.subplots(1, 1, figsize=(7,6))
# fig.subplots_adjust(bottom=0.15, top=0.97, left=0.175, right=0.95)
# ax = axs
# for i, type in enumerate(types):
#     mod, pars = psd(type, mu1=mun[0], mu2=mun[1],  mu3=mun[2])
#     y = mod.eval(pars, x=logr)
#     n_r = y / (4 / 3 * np.pi * r ** 3)
#     ax.plot(r, n_r,  'k',ls=ls[i], lw=2, label=legs[i])
# norm = n_r[r==1]
# ax.plot(r, r**-3*norm,  'r', lw=2)
# ax.plot(r, r**-6*norm,  'g', lw=2)
#
# ax.loglog()
# ax.set_xlim([0.1,100])
# ax.set_ylabel(r'$dN(r)/dlog(r)\ (\mu m^{-1})$')
# ax.set_xlabel(r'$Radius\ (\mu m)$')
# ax.legend()
# plt.show()
# plt.savefig('fig/PSD_dN_dlogr_SPM_number.png',dpi=300)
# #----------------
# # plot dV/dr
# #----------------
#
# fig, axs = plt.subplots(1, 1, figsize=(9,6))
# fig.subplots_adjust(bottom=0.15, top=0.97, left=0.15, right=0.85)
#
# ax = axs
# ax2 = ax.twinx()
# for i, type in enumerate(types):
#     # mod, pars = psd(type)
#     # y = mod.eval(pars, x=logr)
#     # ax.plot(r, y, 'k',ls=ls[i], lw=2,label=legs[i])
#     mod, pars = psd_log(type, mu1=muv[0], mu2=muv[1], mu3=muv[2])
#     v_r = mod.eval(pars, x=r)
#     ax.plot(r, v_r, 'k', ls=ls[i], lw=2, label=legs[i])
#
#
#     mod, pars = psd_log(type,mu1=mun[0], mu2=mun[1],  mu3=mun[2])
#     n_r = mod.eval(pars, x=r)
#     ax2.plot(r[:-1], v_r[:-1]*np.diff(r), 'r', ls=ls[i], lw=2, label=legs[i])
#     #ax.plot(r, n_r, 'g',ls=ls[i], lw=2,label=legs[i])
# # ax.plot(r, (4 / 3 * np.pi * r ** 3)*r**-3,  'r', lw=2)
# # ax.plot(r, (4 / 3 * np.pi * r ** 3)*r**-5,  'g', lw=2)
# ax2.tick_params(axis='y', labelcolor='r')
# ax2.set_ylabel(r'$V(r)\ (\mu m^3\ cm^{-3})$',color='r')
#
# ax.set_ylabel(r'$v(r)\ (\mu m^2\ cm^{-3})$')
# ax.semilogx()
#
# ax.set_xlabel(r'$Radius\ (\mu m)$')
# ax.set_xlim([0.1,100])
# ax.legend()
# plt.savefig('fig/PSD_SPM.png',dpi=300)

