#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:44:36 2019

Code to acces and analyse difference imaged TESS lightcurves from ngtshead

@author: phrhzn
"""

import eleanor
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.table import Table
from TESSselfflatten import TESSflatten
from bls import BLS

def phase_fold_plot(t, lc, period, epoch, target_ID, save_path, title):
    """
    Phase-folds the lc by the given period, and plots a phase-folded light-curve
    for the object of interest
    """
    phase = np.mod(t-epoch-period/2,period)/period 
    
    phase_fold_fig  = plt.figure()
    plt.scatter(phase, lc, c='k', s=2)
    plt.title(title)
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
#    plt.savefig(save_path + '{} - Phase fold plot.png'.format(target_ID))
#    plt.close(phase_fold_fig)

DIAdir = '/ngts/scratch/tess/FFI-LC/S1/clean/'

target_ID = "HIP 105388"
sector = 1

filename = "BANYAN_XI-III_combined_members.csv"   
table_data = Table.read(filename, format='ascii.csv')

# Obtains ra and dec for object from target_ID
i = list(table_data['main_id']).index(target_ID)
ra = table_data['ra'][i]
dec = table_data['dec'][i]
camera = table_data['S1'][i]

star = eleanor.Source(coords=(ra, dec), sector=1)

tic = 79403675
CCD = star.chip
filename = '{}_sector0{}_{}_{}.lc'.format(tic, sector, camera, CCD)

from numpy import loadtxt
lines = loadtxt(filename, delimiter = ' ') # For when in local directory
#lines = loadtxt(DIAdir+filename, delimiter = ' ') # For when on ngtshead
DIA_lc = list(map(list, zip(*lines)))
DIA_mag = np.array(DIA_lc[1])

# Convert TESS magnitudes to flux
DIA_flux =  10**(-0.4*(DIA_mag - 20.60654144))

# Plot Difference imaged data
#plt.figure()
#plt.scatter(DIA_lc[0], DIA_flux/np.median(DIA_flux), s=1, c= 'k')
#plt.ylabel('Normalized Flux')
#plt.xlabel('Time')
#plt.title('{} - Difference imaged light curve from FFIs'.format(target_ID))
#plt.show()

lc = np.vstack((DIA_lc[0], DIA_flux/np.median(DIA_flux), DIA_lc[2])).T

# Run Dave's flattening code
t0 = lc[0,0]
lc[:,0] -= t0
lc[:,1] = TESSflatten(lc,kind='poly', winsize = 3.5, stepsize = 0.15, gapthresh = 0.1)
lc[:,0] += t0

TESSflatten_fig = plt.figure()
TESSflatten_lc = lc[:,1]
plt.scatter(lc[:,0], TESSflatten_lc, c = 'k', s = 1, label = 'TESSflatten flux')
#plt.scatter(p1_times, p1_marker_y, c = 'r', s = 5, label = 'Planet 1')
#plt.scatter(p2_times, p2_marker_y, c = 'g', s = 5, label = 'Planet 2')
plt.title('{} with TESSflatten - Sector {}'.format(target_ID, sector))
plt.ylabel('Normalized Flux')
plt.xlabel('Time - 2457000 [BTJD days]')
#plt.savefig(save_path + '{} - Sector {} - TESSflatten lightcurve.png'.format(target_ID, tpf.sector))
#plt.close(TESSflatten_fig)
plt.show()

durations = np.linspace(0.05, 0.2, 22) * u.day
model = BLS(lc[:,0]*u.day, lc[:,1])
results = model.autopower(durations, frequency_factor=5.0)

# Find the period and epoch of the peak
index = np.argmax(results.power)
period = results.period[index]
t0 = results.transit_time[index]
duration = results.duration[index]
transit_info = model.compute_stats(period, duration, t0)

epoch = transit_info['transit_times'][0]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

# Highlight the harmonics of the peak period
ax.axvline(period.value, alpha=0.4, lw=3)
for n in range(2, 10):
    ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
    ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")

# Plot the periodogram
ax.plot(results.period, results.power, "k", lw=0.5)

ax.set_xlim(results.period.min().value, results.period.max().value)
ax.set_xlabel("period [days]")
ax.set_ylabel("log likelihood")
ax.set_title('{} - BLS Periodogram'.format(target_ID))

phase_fold_plot(lc[:,0]*u.day, lc[:,1], period, epoch, target_ID, save_path = '', title='{} Lightcurve folded by {:0.4} days'.format(target_ID, period))

##################### Plot all four in one for eyeballing #####################
fig, axs = plt.subplots(2,2, figsize = (16,10),  dpi = 120)

# Original DIA data setup
axs[0,0].scatter(DIA_lc[0], DIA_flux/np.median(DIA_flux), s=1, c= 'k')
axs[0,0].set_ylabel('Normalized Flux')
axs[0,0].set_xlabel('Time')
axs[0,0].set_title('{} - Difference imaged light curve from FFIs'.format(target_ID))

# TESSflatten figure setup
axs[0,1].scatter(lc[:,0], TESSflatten_lc, c = 'k', s = 1, label = 'TESSflatten flux')
axs[0,1].set_title('{} with TESSflatten - Sector {}'.format(target_ID, sector))
axs[0,1].set_ylabel('Normalized Flux')
axs[0,1].set_xlabel('Time - 2457000 [BTJD days]')

# Periodogram setup
axs[1,0].plot(results.period, results.power, "k", lw=0.5)
axs[1,0].set_xlim(results.period.min().value, results.period.max().value)
axs[1,0].set_xlabel("period [days]")
axs[1,0].set_ylabel("log likelihood")
axs[1,0].set_title('{} - BLS Periodogram'.format(target_ID))
axs[1,0].axvline(period.value, alpha=0.4, lw=3)
for n in range(2, 10):
    axs[1,0].axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
    axs[1,0].axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")

# Folded plot setup
phase = np.mod(lc[:,0]*u.day-epoch-period/2,period)/period 
axs[1,1].scatter(phase, lc[:,1], c='k', s=2)
axs[1,1].set_title('{} Lightcurve folded by {:0.4} days'.format(target_ID, period))
axs[1,1].set_xlabel('Phase')
axs[1,1].set_ylabel('Normalized Flux')

fig.tight_layout()
plt.show()