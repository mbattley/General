#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:44:12 2019

Reads text files downloaded from NGTS's online viewier

@author: mbattley
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from astropy.table import Table
from utility_belt import phase_fold_plot
from lowess_detrend import lowess_detrending

target_ID = 'HD 33512'

#lines = loadtxt("TYC_7053_832_1.txt", comments="#", delimiter=",", unpack=False)

#data = np.loadtxt('TYC_7053_832_1.txt')
#my_table = Table.read('TYC_7053_832_1.txt')
data = np.load('HD_33512_NGTS_lc_cleaned.npy') 

#print(lines[0])

#data_T = data.transpose()

#time_raw = data_T[0]
#flux_raw = data_T[3]
time_raw = data[0]
flux_raw = data[1]

nancut = np.isnan(time_raw) | np.isnan(flux_raw)
time_raw = time_raw[~nancut]
flux_raw = flux_raw[~nancut]

clean_time = time_raw/(3600*24)
if target_ID == 'TYC 7053-832-1':
    clean_time = time_raw

plt.figure()
plt.scatter(clean_time, flux_raw, c='k', s = 1)

period = 5.528
epoch = 1528.08

# Detrend NGTS flux using lowess detrending
#detrended_TESS_flux, full_lowess_flux = lowess_detrending(time=time_raw,flux=flux_raw,target_ID='HD 33512',n_bins=100)
#flux_TESS = detrended_TESS_flux
full_lowess_flux = np.array([])
lowess = sm.nonparametric.lowess(flux_raw, clean_time, frac=0.008)

#     number of points = 20 at lowest, or otherwise frac = 20/len(t_section) 
#        print(lowess)
overplotted_lowess_full_fig = plt.figure()
plt.scatter(clean_time,flux_raw, c = 'k', s = 1)
plt.plot(lowess[:, 0], lowess[:, 1])
plt.title('{} lc with overplotted lowess full lc detrending'.format(target_ID))
plt.xlabel('Time [BJD days]')
plt.ylabel('Relative flux')
#overplotted_lowess_full_fig.savefig(save_path + "{} lc with overplotted LOWESS full lc detrending.png".format(target_ID))
plt.show()

residual_flux_lowess = flux_raw/lowess[:,1]
full_lowess_flux = np.concatenate((full_lowess_flux,lowess[:,1]))

lowess_full_residuals_fig = plt.figure()
plt.scatter(clean_time,residual_flux_lowess, c = 'k', s = 1)
plt.title('{} lc after lowess full lc detrending'.format(target_ID))
plt.xlabel('Time [BJD days]')
plt.ylabel('Relative flux')
ax = plt.gca()
#ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#            lowess_full_residuals_fig.savefig(save_path + "{} lc after LOWESS full lc detrending.png".format(target_ID))
plt.show()

new_phase, new_lc = phase_fold_plot(clean_time,flux_raw,period,epoch,target_ID,'','{} folded by EB period (5.528d)'.format(target_ID), binned = True)
new_phase, new_lc = phase_fold_plot(clean_time,residual_flux_lowess,period,epoch,target_ID,'','{} folded by EB period (5.528d)'.format(target_ID), binned = True, n_bins=120)