#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:41:34 2019

Code from Laura Kreidberg's batman tutorial and other general batman practice
and transit modelling

@author: phrhzn
"""

import batman
import lightkurve
import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from bls import BLS
from TESSselfflatten import TESSflatten
from astropy.stats import LombScargle
from lightkurve import search_lightcurvefile
from lc_download_methods import *

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
    plt.show()
#    plt.savefig(save_path + '{} - Phase folded by {} days.png'.format(target_ID, period))
#    plt.close(phase_fold_fig)

# Set overall figsize
plt.rcParams["figure.figsize"] = (8.5,4)
plt.rcParams['savefig.dpi'] = 120

save_path = '/home/astro/phrhzn/Documents/PhD/Lightkurve/YSO-BANYAN-targets/Sector-1/Injected transits/Hot Earth/' # On Desktop
flatten = False # defines whether TESSflatten is used later

FG_target_ID_list = ["HIP 1113", "HIP 105388", "HIP 32235", "HD 45270 AB", "HIP 107947", "HIP 116748 A", "HIP 22295", "HD 24636", "HIP 1481"]
KM_target_ID_list = ["RBS 38", "HIP 33737", "AO Men", "AB Dor Aab", "HIP 1993", "AB Pic", "2MASS J23261069-7323498", "2MASS J22424896-7142211", "HIP 107345", "2MASS J20333759-2556521"]
single_target_ID = ["HIP 1113"]
#target_ID = "HIP 22295"

for target_ID in single_target_ID:
    # Import Light-curve of interest
#    with open('Sector_1_target_filenames.pkl', 'rb') as f:
#        target_filenames = pickle.load(f)
#    f.close()
#    
#    if type(target_filenames[target_ID]) == str:
#        filename = target_filenames[target_ID]
#    else:
#        filename = target_filenames[target_ID][0]
#    
#    # Load tpf
#    tpf_30min = lightkurve.search.open(filename)
#    
#    # Attach target name to tpf
#    tpf_30min.targetid = target_ID
#    
#    # Create a median image of the source over time
#    median_image = np.nanmedian(tpf_30min.flux, axis=0)
#    
#    # Select pixels which are brighter than the 85th percentile of the median image
#    aperture_mask = median_image > np.nanpercentile(median_image, 85)
#    
#    # Convert to lightcurve object
#    lc_30min = tpf_30min.to_lightcurve(aperture_mask = aperture_mask).remove_outliers(sigma = 3)
#    #lc_30min = lc_30min[(lc_30min.time < 1346) | (lc_30min.time > 1350)]
#    sigma_cut_lc_fig = lc_30min.scatter().get_figure()
#    plt.title('{} - 30min FFI SAP lc'.format(target_ID))
##    sigma_cut_lc_fig.savefig(save_path + '{} - 3 sigma cut lightcurve.png'.format(target_ID))
#    plt.close(sigma_cut_lc_fig)
    

    ########################### batman stuff ######################################
    type_of_planet = 'Hot Jupiter'
    stellar_type = 'F or G'
    
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = -3.3                        #time of inferior conjunction
    params.per = 8.                      #orbital period (days)
    # Change for type of star
    params.rp = 0.01                       #planet radius (in units of stellar radii)
    params.a = 10.                        #semi-major axis (in units of stellar radii)
    params.inc = 87.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        #limb darkening model
    params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]
    
    lc_30min = diff_image_lc_download('HIP 1113', 1, plot_lc = True)
    
    # Defines times at which to calculate lc and models batman lc
    #t = np.linspace(-13.9165035, 13.9165035, len(lc_30min.time))
    index = int(len(lc_30min.time)//2)
    t = lc_30min.time - lc_30min.time[index]
    m = batman.TransitModel(params, t)
    t += lc_30min.time[index]
    
    batman_flux = m.light_curve(params)
    
#    batman_model_fig = plt.figure()
#    plt.scatter(lc_30min.time, batman_flux, s = 2, c = 'k')
#    plt.xlabel("Time - 2457000 (BTJD days)")
#    plt.ylabel("Relative flux")
#    plt.title("batman model transit for {} around {} Star".format(type_of_planet, stellar_type))
#    batman_model_fig.savefig(save_path + "batman model transit for {} around {} Star".format(type_of_planet,stellar_type))
#    plt.close(batman_model_fig)
    
    ################################# Combining ###################################
    
    combined_flux = np.array(lc_30min.flux)/np.median(lc_30min.flux) + batman_flux -1
    
    injected_transit_fig = plt.figure()
    plt.scatter(lc_30min.time, combined_flux, s = 2, c = 'k')
    plt.xlabel("Time - 2457000 (BTJD days)")
    plt.ylabel("Relative flux")
#    plt.title("{} with injected transits for a {} around a {} Star.".format(target_ID, type_of_planet, stellar_type))
    plt.title("{} with injected transits for a {}R planet to star ratio.".format(target_ID, params.rp))
    ax = plt.gca()
    ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#    injected_transit_fig.savefig(save_path + "{} - Injected transits fig - Period 2 - {} star".format(target_ID, stellar_type))
#    plt.close(injected_transit_fig)
    plt.show()


############################## Removing peaks #################################
    
    flux = combined_flux
    frequency, power = LombScargle(lc_30min.time, flux).autopower()
    i = np.argmax(power)
    freq_rot = frequency[i]
    p_rot = 1/freq_rot
    p_rot = 3.6
    
    t0_rot = 1326
    
    phase = np.mod(t-t0_rot,p_rot)/p_rot
    near_trough = (phase<0.1/p_rot) | (phase>1-0.1/p_rot)
    t_cut_bottom = t[~near_trough]
    flux_cut_bottom = combined_flux[~near_trough]
    flux_err_cut_bottom = lc_30min.flux_err[~near_trough]
    
    phase = np.mod(t_cut_bottom-t0_rot,p_rot)/p_rot
    near_peak = (phase<0.54+0.1/p_rot) & (phase>0.54-0.1/p_rot)
    t_cut = t_cut_bottom[~near_peak]
    flux_cut = flux_cut_bottom[~near_peak]
    flux_err_cut = flux_err_cut_bottom[~near_peak]
    
    # Plot new cut version
    plt.figure()
    plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
    plt.xlabel('Time - 2457000 [BTJD days]')
    plt.ylabel("Relative flux")
    plt.title('{} lc with injected transits after removing peaks/troughs'.format(target_ID))
    ax = plt.gca()
    ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    
############################# Simple line detrending ##########################
    time_diff = np.diff(t_cut)
    residual_flux = np.array([])
    time_from_line_detrend = np.array([])
    
    plt.figure()
    plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
    plt.xlabel('Time - 2457000 [BTJD days]')
    plt.ylabel("Relative flux")
    plt.title('{} lc with injected transits and overplotted n = 3 poly detrending'.format(target_ID))
    
    low_bound = 0
    
    for i in range(len(t_cut)-1):
        if time_diff[i] > 0.1:
            high_bound = i+1
            
            t_section = t_cut[low_bound:high_bound]
            flux_section = flux_cut[low_bound:high_bound]
            z = np.polyfit(t_section, flux_section, 3)
            p = np.poly1d(z)
            plt.plot(t_section, p(t_section), '-')
            
            model_section = p(t_section)
            residuals_section = flux_section/model_section
            residual_flux = np.concatenate((residual_flux,residuals_section))
            time_from_line_detrend = np.concatenate((time_from_line_detrend,t_section))
            low_bound = high_bound
    
    # Carries out same process for final line (up to end of data)        
    high_bound = len(t_cut)
            
    t_section = t_cut[low_bound:high_bound]
    flux_section = flux_cut[low_bound:high_bound]
    z = np.polyfit(t_section, flux_section, 3)
    p = np.poly1d(z)
    plt.plot(t_section, p(t_section), '-')
    
    model_section = p(t_section)
    residuals_section = flux_section/model_section
    residual_flux = np.concatenate((residual_flux,residuals_section))
    time_from_line_detrend = np.concatenate((time_from_line_detrend,t_section))
    
#    t_section = t_cut[83:133]
    plt.figure()
    plt.scatter(time_from_line_detrend,residual_flux, c = 'k', s = 2)
    plt.title('{} lc after n=3 poly detrending'.format(target_ID))
    plt.xlabel('Time - 2457000 [BTJD days]')
    plt.ylabel('Relative flux')
    ax = plt.gca()
    ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')

############################### Simple sine detrending ########################
    
#    flux = combined_flux
#    frequency, power = LombScargle(lc_30min.time, flux).autopower()
#    index = np.argmax(power)
#    guess_freq = frequency[index]
#    
#    A = 0.04
#    w = guess_freq*2*np.pi
#    c = np.mean(flux)
#    p = -1327.2
#    
#    y_model = A * np.sin(w*lc_30min.time + p) + c
#    
#    #res = fit_sin(tt, yynoise)
#    #res = fit_sin(lc_30min.time,flux)
#    #print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )
#    
#    plt.figure()
#    #plt.plot(tt, yy, "-k", label="y", linewidth=2)
#    plt.scatter(lc_30min.time, flux, s=2, c='k', label="y with noise")
#    #plt.plot(lc_30min.time, res["fitfunc"](lc_30min.flux), "r-", label="y fit curve", linewidth=2)
#    plt.plot(lc_30min.time, y_model, "r-", label="y model", linewidth=2)
#    plt.legend(loc="best")
#    plt.show()
#    
#    residuals = flux - y_model
#    
#    plt.figure()
#    plt.scatter(lc_30min.time, residuals, s = 2, c= 'k')
#    
#        # Create periodogram
#    durations = np.linspace(0.05, 0.2, 22) * u.day
##    model = BLS(lc_30min.time*u.day, combined_flux)
#    model = BLS(lc_30min.time*u.day, residuals)
#    results = model.autopower(durations, frequency_factor=5.0)
#    
#    # Find the period and epoch of the peak
#    index = np.argmax(results.power)
#    period = results.period[index]
#    t0 = results.transit_time[index]
#    duration = results.duration[index]
#    transit_info = model.compute_stats(period, duration, t0)
#    
#    epoch = transit_info['transit_times'][0]
#    
#    periodogram_fig, ax = plt.subplots(1, 1, figsize=(8, 4))
#    
#    # Highlight the harmonics of the peak period
#    ax.axvline(period.value, alpha=0.4, lw=3)
#    for n in range(2, 10):
#        ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
#        ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
#    
#    # Plot and save the periodogram
#    ax.plot(results.period, results.power, "k", lw=0.5)
#    ax.set_xlim(results.period.min().value, results.period.max().value)
#    ax.set_xlabel("period [days]")
#    ax.set_ylabel("log likelihood")
#    if flatten == True:
#        ax.set_title('{} - BLS Periodogram with TESSflatten. Injected period = {}'.format(target_ID, params.per))
##        periodogram_fig.savefig(save_path + '{} - BLS Periodogram with TESSflatten - 2 day Per - {}'.format(target_ID, type_of_planet))
#    else:
#        ax.set_title('{} - BLS Periodogram. Injected period = {}'.format(target_ID, params.per))
##        periodogram_fig.savefig(save_path + '{} - BLS Periodogram - {} - 2 day Per'.format(target_ID, type_of_planet))
##    plt.close(periodogram_fig)
#    plt.show()
    
    ########################### Flatten ###########################################
    index = int(len(lc_30min.time)//2)
    lc = np.vstack((t_cut, flux_cut, flux_err_cut)).T
    
    # Run Dave's flattening code
    t0 = lc[0,0]
    
    if flatten == True:
        lc[:,0] -= t0
        lc[:,1] = TESSflatten(lc,kind='poly', winsize = 3.5, stepsize = 0.15, gapthresh = 0.1, polydeg = 3)
        lc[:,0] += t0
    
        TESSflatten_fig = plt.figure()
        TESSflatten_flux = lc[:,1]
        plt.scatter(lc[:,0], TESSflatten_flux, c = 'k', s = 1, label = 'TESSflatten flux')
        #plt.scatter(p1_times, p1_marker_y, c = 'r', s = 5, label = 'Planet 1')
        #plt.scatter(p2_times, p2_marker_y, c = 'g', s = 5, label = 'Planet 2')
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time - 2457000 [BTJD days]')
        ax = plt.gca()
        ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        plt.title('{} with injected transits and TESSflatten - {} day Per'.format(target_ID, params.per))
#        plt.savefig(save_path + '{} - injected {} transit TESSflatten lightcurve - 2day Per.png'.format(target_ID, type_of_planet))
#        plt.close(TESSflatten_fig)
        plt.show()
    
    ########################## Periodogram Stuff ##################################
    
    # Create periodogram
    durations = np.linspace(0.05, 0.2, 22) * u.day
#    model = BLS(lc_30min.time*u.day, combined_flux)
#    model = BLS(lc_30min.time*u.day, lc[:,1])
    if flatten == True:
        BLS_flux = TESSflatten_flux
    else:
        BLS_flux = residual_flux
    model = BLS(t_cut*u.day, BLS_flux)
    results = model.autopower(durations, frequency_factor=5.0)
    
    # Find the period and epoch of the peak
    index = np.argmax(results.power)
    period = results.period[index]
    t0 = results.transit_time[index]
    duration = results.duration[index]
    transit_info = model.compute_stats(period, duration, t0)
    
    epoch = transit_info['transit_times'][0]
    
    periodogram_fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    # Highlight the harmonics of the peak period
    ax.axvline(period.value, alpha=0.4, lw=3)
    for n in range(2, 10):
        ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
    
    # Plot and save the periodogram
    ax.plot(results.period, results.power, "k", lw=0.5)
    ax.set_xlim(results.period.min().value, results.period.max().value)
    ax.set_xlabel("period [days]")
    ax.set_ylabel("log likelihood")
    if flatten == True:
        ax.set_title('{} - BLS Periodogram with TESSflatten after peaks removed. Injected period = {}'.format(target_ID, params.per))
#        periodogram_fig.savefig(save_path + '{} - BLS Periodogram with TESSflatten - 2 day Per - {}'.format(target_ID, type_of_planet))
    else:
        ax.set_title('{} - BLS Periodogram after peaks removed. Injected period = {}'.format(target_ID, params.per))
#        periodogram_fig.savefig(save_path + '{} - BLS Periodogram - {} - 2 day Per'.format(target_ID, type_of_planet))
#    plt.close(periodogram_fig)
    plt.show()    
    
    ################################## Phase folding ##########################
    phase_fold_plot(t_cut, residual_flux, 8, 1328.5, target_ID, '', '{} with injected 8 day transit folded by transit period - {}R ratio'.format(target_ID, params.rp))
    phase_fold_plot(t_cut, residual_flux, 3.6, 1326, target_ID, '', '{} with injected 8 day transit folded by 3.6 day rotation period - {}R ratio'.format(target_ID, params.rp))
    