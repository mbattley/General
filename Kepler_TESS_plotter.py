#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:15:07 2020

@author: mbattley
"""
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)

import lightkurve
import pickle
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
from astropy.table import Table
from utility_belt import binned
from lowess_detrend import lowess_detrending
from lightkurve import search_lightcurvefile



SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

save_path = '/Users/mbattley/Documents/PhD/Kepler-2min xmatch/'

def detrend_with_mask(time,flux,epoch,period,bins=48,pipeline='TESS',target_ID=''):
    duration = 0.1 
    phase = np.mod(time-epoch-period/2,period)/period
    
    near_transit = [False]*len(flux)
    
    for i in range(len(time)):
        if abs(phase[i] - 0.5) < duration/period:
            near_transit[i] = True
    
    near_transit = np.array(near_transit)
    
    t_masked = time[~near_transit]
    flux_masked = flux[~near_transit]
#    flux_err_masked = flux_err_cut[~near_transit]
    t_new = time[near_transit]
    pipeline = 'Kepler'
    if pipeline == 'Kepler':
        f = interpolate.interp1d(t_masked,flux_masked, kind = 'slinear')
    else:
        f = interpolate.interp1d(t_masked,flux_masked, kind = 'quadratic')
#                f = interpolate.BarycentricInterpolator(t_masked,flux_masked)

    flux_new = f(t_new)
    interpolated_fig = plt.figure()
    plt.scatter(t_masked, flux_masked, s = 2, c = 'k')
#    plt.scatter(time_Kepler, flux_Kepler, s = 8, c = 'k')
    plt.scatter(t_new,flux_new, s=8, c = 'r')
    plt.xlabel('Time - 2457000 [BTJD days]')
    plt.ylabel('Relative flux')
#    interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
    
    t_transit_mask = np.concatenate((t_masked,t_new), axis = None)
    flux_transit_mask = np.concatenate((flux_masked,flux_new), axis = None)
    
    sorted_order = np.argsort(t_transit_mask)
    t_transit_mask = t_transit_mask[sorted_order]
    flux_transit_mask = flux_transit_mask[sorted_order]

    #t_cut = lc_30min.time
    #flux_cut = combined_flux
    full_lowess_flux = np.array([])
    lowess = sm.nonparametric.lowess(flux_transit_mask, t_transit_mask, frac=bins/len(time))
    
    overplotted_lowess_full_fig = plt.figure()
    plt.scatter(time,flux, c = 'k', s = 2)
#    plt.plot(lowess[:, 0], lowess[:, 1])
    plt.plot(time, lowess[:, 1])
    plt.title('{} lc with overplotted lowess full lc detrending'.format(target_ID))
    plt.xlabel('Time [BJD]')
    plt.ylabel('Relative flux')
    #overplotted_lowess_full_fig.savefig(save_path + "{} lc with overplotted LOWESS full lc detrending.png".format(target_ID))
    plt.show()
#    plt.close(overplotted_lowess_full_fig)
    
    residual_flux_lowess = flux/lowess[:,1]
    full_lowess_flux = np.concatenate((full_lowess_flux,lowess[:,1]))
    
    lowess_full_residuals_fig = plt.figure()
    plt.scatter(time,residual_flux_lowess, c = 'k', s = 2)
    plt.title('{} lc after lowess full lc detrending'.format(target_ID))
    plt.xlabel('Time [BJD]')
    plt.ylabel('Relative flux')
    ax = plt.gca()
    #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#            lowess_full_residuals_fig.savefig(save_path + "{} lc after LOWESS full lc detrending.png".format(target_ID))
    plt.show()
    
    flux = residual_flux_lowess
    return flux


#table_data = Table.read(save_path + 'Kepler_2min_overlap_observed_with_sectors.csv' , format='ascii.csv')
#
#target_overview_fig = plt.figure()
##fig.set_size_inches(3.54,3.54)
#
#first_15 = True
#first_26 = True
#first_14 = True
#first_multi = True
#
#for i in range(len(table_data['RA'])):
#    if table_data['S15'][i] != 0 and table_data['S26'][i] != 0:
#        if first_multi == True:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='r',s=2, label='S14, S15 & S26')
#            first_multi = False
#        else:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='r',s=2)
#    elif table_data['S15'][i] != 0:
#        if first_15 == True:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='g',s=2, label='S14 & S15')
#            first_15 = False
#        else:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='g',s=2)
#    elif table_data['S26'][i] != 0:
#        if first_26 == True:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='b',s=2, label='S14 & S26')
#            first_26 = False
#        else:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='b',s=2)
#    else:
#        if first_14 == True:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='k',s=2, label='S14 only')
#            first_14 = False
#        else:
#            plt.scatter(table_data['RA'][i],table_data['DEC'][i],c='k',s=2)
#        
#plt.xlabel('RA (deg)')
#plt.ylabel('Dec (deg)')
#handles, labels = plt.gca().get_legend_handles_labels()
#order = [3,0,1,2]
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
##plt.legend()
##plt.xlim(-3, 3)
##plt.ylim(-3, 3)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()


#################### TESS-K2 simulataneous observations #######################

##TESS 2min
#target_ID = 'TIC 4579916'
#sector = 2
#
##    sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = sector, from_file = False)
#lcf = search_lightcurvefile(target_ID, sector=sector).download()
#pdcsap_lc = lcf.PDCSAP_FLUX
#header_0 = lcf
#lc = pdcsap_lc
#nancut = np.isnan(lc.flux) | np.isnan(lc.time)
#lc = lc[~nancut]
#print('Removed nans')
#time_TESS = np.array(lc.time) #n.b. in TJD (TESS Time)
#time_TESS_orig = np.array([float(str(element).strip()) for element in time_TESS]) + 2457000 #Convert to BJD for consistency
#flux_TESS = lc.flux
#flux_TESS_orig = np.array(flux_TESS)/np.median(flux_TESS) 

## K2 directly
#EPIC = 'EPIC 245931711'
#lcfs = lightkurve.search_lightcurvefile(EPIC, mission='K2').download_all()
#stitched_lc = lcfs.PDCSAP_FLUX.stitch()
##stitched_lc = lcfs.PDCSAP_FLUX.stitch(corrector_func=my_custom_corrector_func)
#nancut = np.isnan(stitched_lc.flux) | np.isnan(stitched_lc.time)
#stitched_lc = stitched_lc[~nancut]
#time_K2 = np.array(stitched_lc.time) + 2454833 #Convert to BJD for consistency
#flux_K2 = np.array(stitched_lc.flux)/np.median(stitched_lc.flux) 
#binned_K2_time, binned_K2_flux = binned(time_K2,flux_K2,binsize=2)
#yerr = stitched_lc.flux_err
#
#plt.figure()  
#plt.scatter(time_TESS_orig,flux_TESS_orig, c='r', s=1)
#plt.scatter(time_K2,flux_K2, c='k', s=1)
#plt.xlabel("Time [BJD]")
#plt.ylabel("Normalized Flux [ppt]")
#plt.show()

## K2SFF from download
#filename = 'hlsp_k2sff_k2_lightcurve_245929348-c19_kepler_v1_llc.fits'
#hdul = fits.open(filename)
#k2sff_data = hdul['BESTAPER'].data
#k2sff_time = k2sff_data['T'] + 2454833 #Convert to BJD for consistency
#k2sff_flux = k2sff_data['FCOR']
#
#plt.figure()
#plt.scatter(time_TESS_orig,flux_TESS_orig, c='k', s=1, label = 'TESS 2min')
#plt.scatter(k2sff_time,k2sff_flux,s=2,c='r', label = 'K2-SFF 30min')
#plt.legend()
#plt.xlabel('Time [BJD]')
#plt.ylabel("Normalized Flux")
#plt.show()


########################### PHOT. COMPARISON ##################################

transit_mask - False

with open('HAT_P_7_data.pkl', 'rb') as f:
    kepler_data_reopened = pickle.load(f)
    hat_P_time_TESS = kepler_data_reopened['time_TESS'] 
    hat_P_flux_TESS = kepler_data_reopened['flux_TESS'] +1
    hat_P_time_Kepler = kepler_data_reopened['time_Kepler']
    hat_P_flux_Kepler = kepler_data_reopened['flux_Kepler'] +1
    
with open('Kepler_21_data.pkl', 'rb') as f:
    kepler_data_reopened2 = pickle.load(f)
    k21_time_TESS = kepler_data_reopened2['time_TESS'] 
    k21_flux_TESS = kepler_data_reopened2['flux_TESS'] +1
    k21_time_Kepler = kepler_data_reopened2['time_Kepler']
    k21_flux_Kepler = kepler_data_reopened2['flux_Kepler'] +1

#plt.figure()
#plt.scatter(hat_P_time_Kepler,hat_P_flux_Kepler,s=1)
#plt.show()
#
#plt.figure()
#plt.scatter(hat_P_time_TESS,hat_P_flux_TESS,s=1)
#plt.show()

# Define periods and epochs
# HAT-P_7 b
hat_P_per = 2.20474
hat_P_ep = 2454731.68

# Kepler-21 b
k21_per = 2.78578
k21_ep = 2456798.719

hat_P_flux_Kepler = detrend_with_mask(hat_P_time_Kepler,hat_P_flux_Kepler,hat_P_ep,hat_P_per,bins=48,pipeline='Kepler',target_ID='HAT_P_7')


# Calculate phases
hat_P_phase_Kepler = np.mod(hat_P_time_Kepler-hat_P_ep-hat_P_per/2,hat_P_per)/hat_P_per
hat_P_phase_TESS = np.mod(hat_P_time_TESS-hat_P_ep-hat_P_per/2,hat_P_per)/hat_P_per

k21_phase_Kepler = np.mod(k21_time_Kepler-k21_ep-k21_per/2,k21_per)/k21_per
k21_phase_TESS = np.mod(k21_time_TESS-k21_ep-k21_per/2,k21_per)/k21_per

# Sort phases for binning
sort_hP_Kep = np.argsort(hat_P_phase_Kepler)
hat_P_phase_Kepler = hat_P_phase_Kepler[sort_hP_Kep]
hat_P_flux_Kepler = hat_P_flux_Kepler[sort_hP_Kep]

sort_hP_TESS = np.argsort(hat_P_phase_TESS)
hat_P_phase_TESS = hat_P_phase_TESS[sort_hP_TESS]
hat_P_flux_TESS = hat_P_flux_TESS[sort_hP_TESS]

sort_k21_Kep = np.argsort(k21_phase_Kepler)
k21_phase_Kepler = k21_phase_Kepler[sort_k21_Kep]
k21_flux_Kepler = k21_flux_Kepler[sort_k21_Kep]

sort_k21_TESS = np.argsort(k21_phase_TESS)
k21_phase_TESS = k21_phase_TESS[sort_k21_TESS]
k21_flux_TESS = k21_flux_TESS[sort_k21_TESS]

# bin those boyos
binned_hP_Kepler_phase, binned_hP_Kepler_flux = binned(hat_P_phase_Kepler, hat_P_flux_Kepler, binsize=120)
binned_hP_TESS_phase, binned_hP_TESS_flux = binned(hat_P_phase_TESS, hat_P_flux_TESS, binsize=80)

binned_k21_Kepler_phase, binned_k21_Kepler_flux = binned(k21_phase_Kepler, k21_flux_Kepler, binsize=120)
binned_k21_TESS_phase, binned_k21_TESS_flux = binned(k21_phase_TESS, k21_flux_TESS, binsize=80)

# Plot dat boi
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey='row')

ax1.scatter(hat_P_phase_Kepler,hat_P_flux_Kepler, s=1, c='grey')
ax1.scatter(binned_hP_Kepler_phase, binned_hP_Kepler_flux,marker='o',facecolors='none',s=6,edgecolors='r')
ax2.scatter(hat_P_phase_TESS,hat_P_flux_TESS,s=1, c='k')
ax2.scatter(binned_hP_TESS_phase, binned_hP_TESS_flux,marker='o',facecolors='none',s=6,edgecolors='r')
ax3.scatter(k21_phase_Kepler, k21_flux_Kepler, s=1, c='grey')
ax3.scatter(binned_k21_Kepler_phase, binned_k21_Kepler_flux,marker='o',facecolors='none', s=6, edgecolors='r')
ax4.scatter(k21_phase_TESS, k21_flux_TESS, s=1, c='k')
ax4.scatter(binned_k21_TESS_phase, binned_k21_TESS_flux,marker='o',facecolors='none', s=6, edgecolors='r')
plt.xlim(0,1)
plt.tight_layout()
plt.show()

# Pseudo-code:
# -Load light-curves
# -Mask transits (turn into function)
# -Detrend lcs
# -Fold by known periods of both
# -Plot all four in one plot



#Mask planet in each lc - turn transit masking into a function...
#def transit_mask(time, flux, period, epoch):
#    return
#
#if transit_mask == True:
#    period = periods[planet_num]
#    epoch = t0is[planet_num]
#    duration = 0.4
#    phase = np.mod(time_Kepler-epoch-period/2,period)/period
#    
#    near_transit = [False]*len(flux_Kepler)
#    
#    for i in range(len(time_Kepler)):
#        if abs(phase[i] - 0.5) < duration/period:
#            near_transit[i] = True
#    
#    near_transit = np.array(near_transit)
#    
#    t_masked = time_Kepler[~near_transit]
#    flux_masked = flux_Kepler[~near_transit]
##    flux_err_masked = flux_err_cut[~near_transit]
#    t_new = time_Kepler[near_transit]
#    pipeline = 'Kepler'
#    if pipeline == 'Kepler':
#        f = interpolate.interp1d(t_masked,flux_masked, kind = 'slinear')
#    else:
#        f = interpolate.interp1d(t_masked,flux_masked, kind = 'quadratic')
##                f = interpolate.BarycentricInterpolator(t_masked,flux_masked)
#
#    flux_new = f(t_new)
#    interpolated_fig = plt.figure()
#    plt.scatter(t_masked, flux_masked, s = 2, c = 'k')
##    plt.scatter(time_Kepler, flux_Kepler, s = 8, c = 'k')
#    plt.scatter(t_new,flux_new, s=8, c = 'r')
#    plt.xlabel('Time - 2457000 [BTJD days]')
#    plt.ylabel('Relative flux')
##    interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
#    
#    t_transit_mask = np.concatenate((t_masked,t_new), axis = None)
#    flux_transit_mask = np.concatenate((flux_masked,flux_new), axis = None)
#    
#    sorted_order = np.argsort(t_transit_mask)
#    t_transit_mask = t_transit_mask[sorted_order]
#    flux_transit_mask = flux_transit_mask[sorted_order]

#detrended_TESS_flux, full_lowess_flux = lowess_detrending(time=time_TESS,flux=flux_TESS,target_ID=Kepler_name,n_bins=30)
#flux_TESS = detrended_TESS_flux
#binned_TESS_phase, binned_TESS_flux = binned(phase_TESS, flux_TESS, binsize=15)


