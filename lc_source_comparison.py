#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:24:29 2019

Comparison  between TESS 2min cadence, basic FFI extraction, difference imaging 
and eleanor lightcurves

@author: mbattley
"""

import lightkurve
import eleanor
import pickle
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from lightkurve import search_lightcurvefile
from numpy import loadtxt

# Set overall figsize
plt.rcParams["figure.figsize"] = (8.5,4)
plt.rcParams['savefig.dpi'] = 120

#save_path = '/Users/mbattley/Documents/PhD/Lightkurve/YSO-BANYAN-targets/Sector 1/' # laptop
#save_path = '/home/astro/phrhzn/Documents/PhD/Lightkurve/YSO-BANYAN-targets/Sector 1/' # Desktop
save_path = '/home/astro/phrhzn/Documents/PhD/Comparison of lc pathways/Save-test/' 
#save_path = '/home/u1866052/Plots-for-eyeballing/S1/' # ngtshead

target_ID = "HIP 105388"
sector = 1

filename = "BANYAN_XI-III_members_with_TIC.csv"   
table_data = Table.read(filename, format='ascii.csv')

# Obtains ra and dec for object from target_ID
i = list(table_data['main_id']).index(target_ID)
ra = table_data['ra'][i]
dec = table_data['dec'][i]
camera = table_data['S1'][i]
tic = table_data['MatchID'][i]

########################## Get 2min cadence lc- SAP flux ######################

lcf = search_lightcurvefile(target_ID, sector = 1).download()

# overplot both SAP and PDCSAP lightcurves
lcf.plot()
combined_2min = plt.gcf()
combined_2min.savefig(save_path + '{} - Sector {} - Combined 2min SAP and PDCSAP fluxes.png'.format(target_ID, sector))
plt.show()

# Seperate lightcurves
sap_lc = lcf.SAP_FLUX
pdcsap_lc = lcf.PDCSAP_FLUX

# Plot SAP lc
SAP_2min = sap_lc.bin(binsize = 15).scatter().get_figure()
plt.title('{} - 2min lc - SAP flux'.format(target_ID))
SAP_2min.savefig(save_path + '{} - Sector {} - 2min SAP flux.png'.format(target_ID, sector))
plt.show()

# Plot PDC lc
PDCSAP_2min = pdcsap_lc.bin(binsize = 15).scatter().get_figure()
plt.title('{} - 2min lc - PDCSAP flux'.format(target_ID))
PDCSAP_2min.savefig(save_path + '{} - Sector {} - 2min PDCSAP flux.png'.format(target_ID, sector))
plt.show

########################### FFI base lc ########################################
#
#with open('Sector_1_target_filenames.pkl', 'rb') as f:
#    target_filenames = pickle.load(f)
#f.close()
#
#if type(target_filenames[target_ID]) == str:
#    filename = target_filenames[target_ID]
#else:
#    filename = target_filenames[target_ID][0]
#
## Load tpf
#tpf_30min = lightkurve.search.open(filename)
#
## Attach target name to tpf
#tpf_30min.targetid = target_ID
#
## Create a median image of the source over time
#median_image = np.nanmedian(tpf_30min.flux, axis=0)
#
## Select pixels which are brighter than the 85th percentile of the median image
#aperture_mask = median_image > np.nanpercentile(median_image, 85)
#
## Plot and save tpf
#tpf_30min.plot(aperture_mask = aperture_mask)
##tpf_plot.savefig(save_path + '{} - Sector {} - tpf plot.png'.format(target_ID, tpf.sector))
##plt.close(tpf_plot)
#
## Remove plot base lc
#tpf_30min.to_lightcurve(aperture_mask = aperture_mask).plot()
##sigma_cut_lc_fig.savefig(save_path + '{} - Sector {} - 3 sigma lightcurve.png'.format(target_ID, tpf.sector))
##plt.close(sigma_cut_lc_fig)
#
## Convert to lightcurve object
#lc_30min = tpf_30min.to_lightcurve(aperture_mask = aperture_mask)
#lc_30min = lc_30min[(lc_30min.time < 1346) | (lc_30min.time > 1350)]
#lc_30min.scatter()
#plt.title('{} - 30min FFI base lc'.format(target_ID))
#
############################# eleanor lc #######################################

star = eleanor.Source(coords=(ra, dec), sector=1)
#star = eleanor.Source(coords=(49.4969, -66.9268), sector=1)

# Extract target pixel file, perform aperture photometry and complete some systematics corrections
data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False)

q = data.quality == 0

# Plot raw flux
raw_eleanor_fig = plt.figure()
plt.scatter(data.time[q], data.raw_flux[q]/np.median(data.raw_flux[q]), s=1, c = 'k')
plt.ylabel('Normalized Flux')
plt.xlabel('Time')
plt.title('{} - eleanor light curve from FFIs - raw flux'.format(target_ID))
raw_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor raw flux.png'.format(target_ID, sector))
plt.show()

# Plot corrected flux
corr_eleanor_fig = plt.figure()
plt.scatter(data.time[q], data.corr_flux[q]/np.median(data.corr_flux[q]), s=1, c= 'r')
plt.ylabel('Normalized Flux')
plt.xlabel('Time')
plt.title('{} - eleanor light curve from FFIs - corr flux'.format(target_ID))
corr_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor corr flux.png'.format(target_ID, sector))
plt.show()

# Plot pca flux
eleanor.TargetData.pca(data, flux=data.raw_flux, modes=4)
pca_eleanor_fig = plt.figure()
plt.scatter(data.time[q], data.pca_flux[q]/np.median(data.pca_flux[q]), s=1, c= 'g')
plt.ylabel('Normalized Flux')
plt.xlabel('Time')
plt.title('{} - eleanor light curve from FFIs - pca flux'.format(target_ID))
pca_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor pca flux.png'.format(target_ID, sector))
plt.show()

# Plot psf flux
#eleanor.TargetData.psf_lightcurve(data, model='gaussian', likelihood='poisson')
#psf_eleanor_fig = plt.figure()
#plt.scatter(data.time[q], data.psf_flux[q]/np.median(data.psf_flux[q]), s=1, c= 'g')
#plt.ylabel('Normalized Flux')
#plt.xlabel('Time')
#plt.title('{} - eleanor light curve from FFIs - psf flux'.format(target_ID))
#psf_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor psf flux.png'.format(target_ID, sector))
#plt.show()


########################## Difference imaging lc ##############################

DIAdir = '/ngts/scratch/tess/FFI-LC/S1/clean/'

CCD = star.chip
filename = '{}_sector0{}_{}_{}.lc'.format(tic, sector, camera, CCD)

try:
    lines = loadtxt(filename, delimiter = ' ') # For when in local directory
#    lines = loadtxt(DIAdir+filename, delimiter = ' ') # For when on ngtshead
    DIA_lc = list(map(list, zip(*lines)))
    DIA_mag = np.array(DIA_lc[1])
    
    # Convert TESS magnitudes to flux
    DIA_flux =  10**(-0.4*(DIA_mag - 20.60654144))
    
    # Plot Difference imaged data
    diffimage_fig = plt.figure()
    plt.scatter(DIA_lc[0], DIA_flux/np.median(DIA_flux), s=1, c= 'k')
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time')
    plt.title('{} - Difference imaged light curve from FFIs'.format(target_ID))
    diffimage_fig.savefig(save_path + '{} - Sector {} - DiffImage flux.png'.format(target_ID, sector))
    plt.show()
except:
    print('The file {} does not exist - difference imaging data not available for {}'.format(filename,target_ID))
# Note: Might be easiest to make all automatically save and define figure size for easiest comparison...
