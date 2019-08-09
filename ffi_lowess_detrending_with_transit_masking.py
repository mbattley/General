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
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import statsmodels.api as sm
from TESSselfflatten import TESSflatten
from astropy.timeseries import LombScargle
from lightkurve import search_lightcurvefile
from lc_download_methods import *
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import find_peaks
from astropy.timeseries import BoxLeastSquares
from wotan import flatten
from astropy.io import ascii
from astropy.table import Table
from scipy import interpolate

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
    plt.savefig(save_path + '{} - Phase folded by {} days.png'.format(target_ID, period))
    plt.show(block = False)
    #plt.close(phase_fold_fig)
 
 
def bin(time, flux, binsize=13, method='mean'):
    """Bins a lightcurve in blocks of size `binsize`.
    n.b. based on the one from eleanor

    The value of the bins will contain the mean (`method='mean'`) or the
    median (`method='median'`) of the original data.  The default is mean.

    Parameters
    ----------
    binsize : int
        Number of cadences to include in every bin.
    method: str, one of 'mean' or 'median'
        The summary statistic to return for each bin. Default: 'mean'.

    Returns
    -------
    binned_lc : LightCurve object
        Binned lightcurve.

    Notes
    -----
    - If the ratio between the lightcurve length and the binsize is not
      a whole number, then the remainder of the data points will be
      ignored.
    """
    available_methods = ['mean', 'median']
    if method not in available_methods:
        raise ValueError("method must be one of: {}".format(available_methods))
    methodf = np.__dict__['nan' + method]

    n_bins = len(flux) // binsize
    indexes = np.array_split(np.arange(len(time)), n_bins)
    binned_time = np.array([methodf(time[a]) for a in indexes])
    binned_flux = np.array([methodf(flux[a]) for a in indexes])

    return binned_time, binned_flux

# Set overall figsize
plt.rcParams["figure.figsize"] = (8.5,4)
plt.rcParams['savefig.dpi'] = 180


########################## INPUTS #####################################################
#save_path = '/home/astro/phrhzn/Documents/PhD/Lowess detrending/TESS S1/' # On Desktop
#save_path = '/home/u1866052/Lowess detrending/TESS S2/Reanalysed/' # ngtshead
save_path = '/Users/mbattley/Documents/PhD/New detrending methods/Smoothing/lowess/TESS S1/' # On laptop
sector = 1
use_TESSflatten = False # defines whether TESSflatten is used later
use_peak_cut = False
binned = False
transit_mask = True
detrending = 'lowess_partial' # Can be 'poly', 'lowess_full', 'lowess_partial', 'TESSflatten', 'wotan' OR 'None'
single_target_ID = ["HIP 116748 A"]
######################################################################################

# Set up table to collect all info on any periodic main stellar variability
variability_table = Table({'Name':[],'LS_Period':[],'BLS_Period':[],'Var_Amplitude':[]},names=['Name','LS_Period','BLS_Period','Var_Amplitude'])
variability_table['Name'] = variability_table['Name'].astype(str)

# Other Possible target lists
FG_target_ID_list = ["HIP 1113", "HIP 105388", "HIP 32235", "HD 45270 AB", "HIP 107947", "HIP 116748 A", "HIP 22295", "HD 24636", "HIP 1481"]
KM_target_ID_list = ["RBS 38", "HIP 33737", "AO Men", "AB Dor Aab", "HIP 1993", "AB Pic", "2MASS J23261069-7323498", "2MASS J22424896-7142211", "HIP 107345", "2MASS J20333759-2556521"]
all_target_IDs = ["2MASS J02235464-5815067","AK Pic AB","CD-60 416","CD-61 6893","GJ 907.1","HD 10269","HD 14228 A","HD 203","HD 20888","HD 217343","HD 223352 AB","HD 24636","HD 45270 AB","HIP 12394","HIP 490","HIP 9902","J0024-4053","J0148-4831","J0224-7633","J0249-6228","J0315-7723","J0416-5841","J0427-7719","J0436-6001","J0517-6634","J0524-7109","J0535-7053","J0536-6555","J0605-6559","J0610-6129","J0618-5645","J0624-6948","J0640-7051","J0658-6311","J0700-6203","J0703-5505","J0703-6110","J0811-6656","J0839-6954","J0841-7113","J0842-7113","J2318-4049","J2348-2807","TYC 8534-1810-1"]
#target_ID = "HIP 22295"
#with open('Sector_3_targets_from_TIC_list.pkl', 'rb') as f:
#    sector_3_targets = pickle.load(f)
#sector_1_targets = ['AB Pic', 'J0535-7053', 'HD 20888', 'J0346-6246', 'J0120-6241', 'HIP 22295', 'J0413-8408', 'J0249-8421', 'J0519-7104', 'J0350-6949', 'J0524-7109', 'J0538-7413', 'HD 24636', 'WOH S 216', 'J0524-7038', 'HD 45270 AB', 'J0536-6555', 'TYC 8881-551-1', 'J0608-8133', 'J0247-6808', 'J0640-7051', 'J0249-6228', 'HIP 32235', 'J0425-7630', 'J0427-7719', 'HD 42270', 'HIP 12394', 'J0224-7633', 'TYC 8896-340-1', 'AO Men', '2MASS J23261069-7323498', 'HIP 116748 A', 'HIP 116748 B', '2MASS J20333759-2556521', 'HIP 1113', 'HIP 107947', 'J0101-7250', 'HIP 107345', 'J2319-4748', '2MASS J01231125-6921379', 'HIP 1993', 'RBS 38', '2MASS J22424896-7142211', 'J0156-7457', 'HIP 105388', 'J2158-7048', 'HIP 1481', 'CD-61 6893', 'J0315-7723', 'HD 207043', 'J0608-5703', 'AB Dor Aab', 'J2158-4705', 'J0820-6247', 'J0226-6700', 'J2146-2515', 'PSO J318.5-22', 'J0804-6243', 'J0501-7856', 'L 106-104', 'AT Mic B', 'HD 20888', 'HIP 116748 B']

for target_ID in single_target_ID:
    try:
        try:
            lc_30min, filename = diff_image_lc_download(target_ID, sector, plot_lc = True, save_path = save_path)
            pipeline = 'Diff_Image'
        except:
        		#print('Lightcurve for {} not available'.format(target_ID))
            try:
                raw_lc, corr_lc, pca_lc = eleanor_lc_download(target_ID, sector, from_file = True, save_path = save_path, plot_pca = False)
                lc_30min = pca_lc
                pipeline = 'eleanor'
            except RuntimeError:
                print('Lightcurve for {} not available'.format(target_ID))
#        sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector)
#        lc_30min = pdcsap_lc
#        pipeline = '2min'

    
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
    
        ######################### Find rotation period ################################
        normalized_flux = np.array(lc_30min.flux)/np.median(lc_30min.flux)
        
        # From Lomb-Scargle
        freq = np.arange(0.04,4.1,0.00001)
        power = LombScargle(lc_30min.time, normalized_flux).power(freq)
        ls_fig = plt.figure()
        plt.plot(freq, power, c='k', linewidth = 1)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('{} LombScargle Periodogram for original lc'.format(target_ID))
        #ls_plot.show(block=True)
#        ls_fig.savefig(save_path + '{} - Lomb-Sacrgle Periodogram for original lc.png'.format(target_ID))
        plt.close(ls_fig)
        i = np.argmax(power)
        freq_rot = freq[i]
        p_rot = 1/freq_rot
        print('Rotation Period = {:.3f}d'.format(p_rot))
        
        # From BLS
        durations = np.linspace(0.05, 1, 100) * u.day
        model = BoxLeastSquares(lc_30min.time*u.day, normalized_flux)
#        model = BLS(lc_30min.time*u.day, BLS_flux)
        results = model.autopower(durations, frequency_factor=5.0)
        rot_index = np.argmax(results.power)
        rot_period = results.period[rot_index]
        rot_t0 = results.transit_time[rot_index]
        print("Rotation Period from BLS of original = {}d".format(rot_period))
        
        ########################### batman stuff ######################################
  #      type_of_planet = 'Hot Jupiter'
   #     stellar_type = 'F or G'
        #params = batman.TransitParams()       #object to store transit parameters
        #print("batman works y'all")
        #params.t0 = -4.5                      #time of inferior conjunction
        #params.per = 8.0                      #orbital period (days) - try 0.5, 1, 2, 4, 8 & 10d periods
        # Change for type of star
        #params.rp = 0.05                      #planet radius (in units of stellar radii) - Try between 0.01 and 0.1 (F/G) or 0.025 to 0.18 (K/M)
        # For a: 25 for 10d; 17 for 8d; 10 for 4d; 4-8 (6) for 2 day; 2-5  for 1d; 1-3 (or 8?) for 0.5d
        #params.a = 17.                         #semi-major axis (in units of stellar radii) - 10-20 probably most realistic for 4 or 8 day; 4-8 for 2 day; 2-5 for 1d; 1-3 for 0.5d
        #params.inc = 87.                      #orbital inclination (in degrees)
        #params.ecc = 0.                       #eccentricity
        #params.w = 90.                        #longitude of periastron (in degrees)
        #params.limb_dark = "nonlinear"        #limb darkening model
        #params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]
        #print("Finished building params")
        
#    #    try:
#    #        lc_30min, filename = diff_image_lc_download(target_ID, 1, plot_lc = True)
#    #    except:
#    #        break
#        
     #   # Defines times at which to calculate lc and models batman lc
        #t = np.linspace(-13.9165035, 13.9165035, len(lc_30min.time))
        #index = int(len(lc_30min.time)//2)
        #mid_point = lc_30min.time[index]
        #t = lc_30min.time - lc_30min.time[index]
        #m = batman.TransitModel(params, t)
        #t += lc_30min.time[index]
        #print("About to compute flux")
        #batman_flux = m.light_curve(params)
        #print("Computed flux")
        #batman_model_fig = plt.figure()
        #plt.scatter(lc_30min.time, batman_flux, s = 2, c = 'k')
        #plt.xlabel("Time - 2457000 (BTJD days)")
        #plt.ylabel("Relative flux")
        #plt.title("batman model transit for {}R ratio".format(params.rp))
        #batman_model_fig.savefig(save_path + "batman model transit for {} around {} Star".format(type_of_planet,stellar_type))
        #plt.close(batman_model_fig)
        #plt.show()
        
        ################################# Combining ###################################
        
        #combined_flux = np.array(lc_30min.flux)/np.median(lc_30min.flux) + batman_flux -1
        
        #injected_transit_fig = plt.figure()
        #plt.scatter(lc_30min.time, combined_flux, s = 2, c = 'k')
        #plt.xlabel("Time - 2457000 (BTJD days)")
        #plt.ylabel("Relative flux")
    #    plt.title("{} with injected transits for a {} around a {} Star.".format(target_ID, type_of_planet, stellar_type))
        #plt.title("{} with injected transits for a {}R planet to star ratio.".format(target_ID, params.rp))
        #ax = plt.gca()
        #for n in range(int(-1*8/params.per),int(2*8/params.per+2)):
        #    ax.axvline(params.t0+n*params.per+mid_point, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #injected_transit_fig.savefig(save_path + "{} - Injected transits fig - Period 8 - {}R transit.png".format(target_ID, params.rp))
#        plt.close(injected_transit_fig)
        #plt.show()
    
    ############################## Removing peaks #################################
        
        combined_flux = np.array(lc_30min.flux)/np.median(lc_30min.flux)
        if use_peak_cut == True:
            peaks, peak_info = find_peaks(combined_flux, prominence = 0.001, width = 8)
            #peaks = np.array([64, 381, 649, 964, 1273])
            troughs, trough_info = find_peaks(-combined_flux, prominence = -0.001, width = 8)
            #troughs = np.array([211, 530, 795, 1113])
            #troughs = np.append(troughs, [370,1031])
            #print(troughs)
            flux_peaks = combined_flux[peaks]
            flux_troughs = combined_flux[troughs]
            amplitude_peaks = ((flux_peaks[0]-1) + (1-flux_troughs[0]))/2
            print("Absolute amplitude of main variability = {}".format(amplitude_peaks))
            peak_location_fig = plt.figure()
            plt.scatter(lc_30min.time, combined_flux, s = 2, c = 'k')
            plt.plot(lc_30min.time[peaks], combined_flux[peaks], "x")
            plt.plot(lc_30min.time[troughs], combined_flux[troughs], "x", c = 'r')
#            peak_location_fig.savefig(save_path + "{} - Peak location fig.png".format(target_ID))
            peak_location_fig.show()
            #plt.close(peak_location_fig)
            
            near_peak_or_trough = [False]*len(combined_flux)
            
            for i in peaks:
                for j in range(len(lc_30min.time)):
                    if abs(lc_30min.time[j] - lc_30min.time[i]) < 0.1:
                        near_peak_or_trough[j] = True
            
            for i in troughs:
                for j in range(len(lc_30min.time)):
                    if abs(lc_30min.time[j] - lc_30min.time[i]) < 0.1:
                        near_peak_or_trough[j] = True
            
            near_peak_or_trough = np.array(near_peak_or_trough)
            
            t_cut = lc_30min.time[~near_peak_or_trough]
            flux_cut = combined_flux[~near_peak_or_trough]
            flux_err_cut = lc_30min.flux_err[~near_peak_or_trough]
        #    
        #    phase = np.mod(t-t0_rot,p_rot)/p_rot
        #    plt.figure()
        #    plt.scatter(phase,flux, c = 'k', s = 2)
        #    near_trough = (phase<0.1/p_rot) | (phase>1-0.1/p_rot)
        #    t_cut_bottom = t[~near_trough]
        #    flux_cut_bottom = combined_flux[~near_trough]
        #    flux_err_cut_bottom = lc_30min.flux_err[~near_trough]
        #    
        #    phase = np.mod(t_cut_bottom-t0_rot,p_rot)/p_rot
        #    near_peak = (phase<0.5+0.1/p_rot) & (phase>0.5-0.1/p_rot)
        #    t_cut = t_cut_bottom[~near_peak]
        #    flux_cut = flux_cut_bottom[~near_peak]
        #    flux_err_cut = flux_err_cut_bottom[~near_peak]
        #    
        #    cut_phase = np.mod(t_cut-t0_rot,p_rot)/p_rot
        #    plt.figure()
        #    plt.scatter(cut_phase, flux_cut, c='k', s=2)
        #    
            # Plot new cut version
            peak_cut_fig = plt.figure()
            plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel("Relative flux")
            plt.title('{} lc after removing peaks/troughs'.format(target_ID))
            ax = plt.gca()
            #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            peak_cut_fig.savefig(save_path + "{} - Peak cut fig.png".format(target_ID))
            #peak_cut_fig.show()
            plt.close(peak_cut_fig)
        else:
             t_cut = lc_30min.time
             flux_cut = combined_flux
             flux_err_cut = lc_30min.flux_err
             print('Flux cut skipped')
             
    ############################## Apply transit mask #########################

        if transit_mask == True:
            period = 8.138268
            epoch = 1332.30997
            duration = 0.08
            phase = np.mod(t_cut-epoch-period/2,period)/period
            
            near_transit = [False]*len(flux_cut)
            
            for i in range(len(t_cut)):
                if abs(phase[i] - 0.5) < duration/period:
                    near_transit[i] = True
            
            near_transit = np.array(near_transit)
            
            t_masked = t_cut[~near_transit]
            flux_masked = flux_cut[~near_transit]
            flux_err_masked = flux_err_cut[~near_transit]
            
            f = interpolate.interp1d(t_masked,flux_masked)
            t_new = t_cut[near_transit]
            flux_new = f(t_new)
            interpolated_fig = plt.figure()
            plt.scatter(t_masked, flux_masked, s = 2, c = 'k')
            plt.scatter(t_new,flux_new, s=2, c = 'r')
#            interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
            
            t_transit_mask = np.concatenate((t_masked,t_new), axis = None)
            flux_transit_mask = np.concatenate((flux_masked,flux_new), axis = None)
            
            sorted_order = np.argsort(t_transit_mask)
            t_transit_mask = t_transit_mask[sorted_order]
            flux_transit_mask = flux_transit_mask[sorted_order]
            
         
    #################################### Wotan ####################################
        if detrending == 'wotan':
            flatten_lc_before, trend_before = flatten(lc_30min.time, combined_flux, window_length=0.3, method='hspline', return_trend = True)
            if transit_mask == True:
                flatten_lc_after, trend_after = flatten(t_transit_mask, flux_transit_mask, window_length=0.3, method='hspline', return_trend = True)
            else:
                flatten_lc_after, trend_after = flatten(t_cut, flux_cut, window_length=0.3, method='hspline', return_trend = True)
            
            # Plot before peak removal
            wotan_original_lc_fig = plt.figure()
            plt.scatter(lc_30min.time,flatten_lc_before, c = 'k', s = 2)
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel("Relative flux")
            plt.title('{} lc after standard wotan detrending - before peak removal'.format(target_ID))
            wotan_original_lc_fig.savefig(save_path + "{} lc residuals after wotan detrending of original lc".format(target_ID))
            wotan_original_lc_fig.show()
            #plt.close(overplotted_lowess_full_fig)
            
            # Plot after peak removal
            wotan_peak_removed_fig = plt.figure()
            plt.scatter(t_cut,flatten_lc_after, c = 'k', s = 2)
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel("Relative flux")
            plt.title('{} lc after standard wotan detrending - after peak removal'.format(target_ID))
            wotan_peak_removed_fig.savefig(save_path + "{} residuals after peak removal and wotan detrending".format(target_ID))
            wotan_peak_removed_fig.show()
            #plt.close(overplotted_lowess_full_fig)
            
            # Plot wotan detrending over data
            overplotted_wotan_fig = plt.figure()
            plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
            plt.plot(t_cut, trend_after)
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel("Relative flux")
            plt.title('{} lc with injected transits and overplotted wotan detrending'.format(target_ID))
            overplotted_wotan_fig.savefig(save_path + "{} lc with overplotted wotan detrending".format(target_ID))
            overplotted_wotan_fig.show()
            #plt.close(overplotted_lowess_full_fig)
    
    
    ############################## LOWESS detrending ##############################
        
        # Full lc
        if detrending == 'lowess_full':
            #t_cut = lc_30min.time
            #flux_cut = combined_flux
            if transit_mask == True:
                lowess = sm.nonparametric.lowess(flux_transit_mask, t_transit_mask, frac=0.02)
            else:
                lowess = sm.nonparametric.lowess(flux_cut, t_cut, frac=0.02)
            
        #     number of points = 20 at lowest, or otherwise frac = 20/len(t_section) 
            
            overplotted_lowess_full_fig = plt.figure()
            plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
            plt.plot(lowess[:, 0], lowess[:, 1])
            plt.title('{} lc with overplotted lowess full lc detrending'.format(target_ID))
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel('Relative flux')
            overplotted_lowess_full_fig.savefig(save_path + "{} lc with overplotted LOWESS full lc detrending.png".format(target_ID))
            plt.show()
            #plt.close(overplotted_lowess_full_fig)
            
            residual_flux_lowess = flux_cut/lowess[:,1]
            
            lowess_full_residuals_fig = plt.figure()
            plt.scatter(t_cut,residual_flux_lowess, c = 'k', s = 2)
            plt.title('{} lc after lowess full lc detrending'.format(target_ID))
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel('Relative flux')
            ax = plt.gca()
            #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#            lowess_full_residuals_fig.savefig(save_path + "{} lc after LOWESS full lc detrending.png".format(target_ID))
            plt.show()
            #plt.close(lowess_full_residuals_fig)
            
            
        # Partial lc
        if detrending == 'lowess_partial':
            time_diff = np.diff(t_cut)
            residual_flux_lowess = np.array([])
            time_from_lowess_detrend = np.array([])
            
            overplotted_detrending_fig = plt.figure()
            plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel("Relative flux")
            plt.title('{} lc with overplotted LOWESS partial lc detrending'.format(target_ID))
            
            low_bound = 0
            
            for i in range(len(t_cut)-1):
                if time_diff[i] > 0.1:
                    high_bound = i+1
                    
                    t_section = t_cut[low_bound:high_bound]
                    flux_section = flux_cut[low_bound:high_bound]
                    if transit_mask == True:
                        lowess = sm.nonparametric.lowess(flux_transit_mask[low_bound:high_bound], t_transit_mask[low_bound:high_bound], frac=20/len(t_section))
                    else:
                        lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
#                    lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
                    lowess_flux_section = lowess[:,1]
                    plt.plot(t_section, lowess_flux_section, '-')
                    
                    residuals_section = flux_section/lowess_flux_section
                    residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
                    time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
                    low_bound = high_bound
            
            # Carries out same process for final line (up to end of data)        
            high_bound = len(t_cut)
                    
            t_section = t_cut[low_bound:high_bound]
            flux_section = flux_cut[low_bound:high_bound]
            if transit_mask == True:
                lowess = sm.nonparametric.lowess(flux_transit_mask[low_bound:high_bound], t_transit_mask[low_bound:high_bound], frac=20/len(t_section))
            else:
                lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
#            lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
            lowess_flux_section = lowess[:,1]
            plt.plot(t_section, lowess_flux_section, '-')
#            overplotted_detrending_fig.savefig(save_path + "{} - Overplotted lowess detrending - partial lc.png".format(target_ID))
            overplotted_detrending_fig.show()
            #plt.close(overplotted_detrending_fig)
            
            residuals_section = flux_section/lowess_flux_section
            residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
            time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
            
        #    t_section = t_cut[83:133]
            residuals_after_lowess_fig = plt.figure()
            plt.scatter(time_from_lowess_detrend,residual_flux_lowess, c = 'k', s = 2)
            plt.title('{} lc after LOWESS partial lc detrending'.format(target_ID))
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel('Relative flux')
            #ax = plt.gca()
            #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#            residuals_after_lowess_fig.savefig(save_path + "{} lc after LOWESS partial lc detrending.png".format(target_ID))
            residuals_after_lowess_fig.show()
            #plt.close(residuals_after_lowess_fig)
    
        
    ########################### TESSflatten ###########################################
        if use_TESSflatten == True:
            index = int(len(lc_30min.time)//2)
            #lc = np.vstack((t_cut, flux_cut, flux_err_cut)).T
            #lc = np.vstack((t_cut, residual_flux, flux_err_cut)).T
            lc = np.vstack((lc_30min.time, combined_flux, lc_30min.flux_err)).T
            print('lc built fine')
            # Run Dave's flattening code
            t0 = lc[0,0]
            lc[:,0] -= t0
            lc[:,1] = TESSflatten(lc,kind='poly', winsize = 3.5, stepsize = 0.15, gapthresh = 0.1, polydeg = 3)
            lc[:,0] += t0
            print('TESSflatten used')
            TESSflatten_fig = plt.figure()
            TESSflatten_flux = lc[:,1]
            plt.scatter(lc[:,0], TESSflatten_flux, c = 'k', s = 1, label = 'TESSflatten flux')
            #plt.scatter(p1_times, p1_marker_y, c = 'r', s = 5, label = 'Planet 1')
            #plt.scatter(p2_times, p2_marker_y, c = 'g', s = 5, label = 'Planet 2')
            plt.ylabel('Normalized Flux')
            plt.xlabel('Time - 2457000 [BTJD days]')
            #ax = plt.gca()
            #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            plt.title('{} with TESSflatten'.format(target_ID))
            if binned == True:
            	binned_time, binned_flux = bin(lc[:,0], TESSflatten_flux)
            	plt.plot(binned_time, binned_flux, c = 'r', label = 'TESSflatten flux')
            TESSflatten_fig.savefig(save_path + '{} - TESSflatten lightcurve.png'.format(target_ID))
            #plt.close(TESSflatten_fig)
            print('TESSflatten Plotted')
            TESSflatten_fig.show()
 
        
    #    ########################## Periodogram Stuff ##################################
        
        # Create periodogram
        durations = np.linspace(0.05, 0.5, 100) * u.day
        if use_TESSflatten == True:
            BLS_flux = TESSflatten_flux
        elif detrending == 'lowess_full' or detrending == 'lowess_partial':
            BLS_flux = residual_flux_lowess
        elif detrending == 'wotan':
            BLS_flux = flatten_lc_after
        else:
            BLS_flux = combined_flux
        model = BoxLeastSquares(t_cut*u.day, BLS_flux)
        #model = BLS(lc_30min.time*u.day,BLS_flux)
        results = model.autopower(durations, minimum_n_transit=3,frequency_factor=1.0)
        
        # Find the period and epoch of the peak
        index = np.argmax(results.power)
        period = results.period[index]
        #print(results.period)
        t0 = results.transit_time[index]
        duration = results.duration[index]
        transit_info = model.compute_stats(period, duration, t0)
        print(transit_info)
        
        epoch = transit_info['transit_times'][0]
        
    #    periodogram_fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        periodogram_fig, ax = plt.subplots(1, 1)
        
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
        if use_TESSflatten == True:
            ax.set_title('{} - BLS Periodogram with TESSflatten'.format(target_ID))
#            periodogram_fig.savefig(save_path + '{} - BLS Periodogram with TESSflatten.png'.format(target_ID))
        else:
            ax.set_title('{} - BLS Periodogram after {} detrending'.format(target_ID, detrending))
#            periodogram_fig.savefig(save_path + '{} - BLS Periodogram after lowess partial detrending.png'.format(target_ID))
        #plt.close(periodogram_fig)
        periodogram_fig.show()   
    	  
    
    ##    ################################## Phase folding ##########################
        #phase_fold_plot(t_cut, BLS_flux, 8, mid_point+params.t0, target_ID, save_path, '{} with injected 8 day transit folded by transit period - {}R ratio'.format(target_ID, params.rp))
        #phase_fold_plot(lc_30min.time, BLS_flux, rot_period.value, rot_t0.value, target_ID, save_path, '{} folded by rotation period'.format(target_ID))
        #print('Max BLS Period = {} days, t0 = {}'.format(period.value, t0.value))        
        phase_fold_plot(t_cut, BLS_flux, period.value, t0.value, target_ID, save_path, '{} {} residuals folded by Periodogram Max ({:.3f} days)'.format(target_ID, detrending, period.value))
        period_to_test = p_rot
        t0_to_test = 1332.30997
        period_to_test2 = 8.138268
        t0_to_test2 = 1332.30997
#        period_to_test3 = 10.9
#        t0_to_test3 = 1339
#        period_to_test4 = 13.45
#        t0_to_test4 = 1339          
        phase_fold_plot(t_cut, BLS_flux, p_rot, t0_to_test, target_ID, save_path, '{} folded by rotation period ({} days)'.format(target_ID,period_to_test))
        phase_fold_plot(t_cut, BLS_flux, period_to_test2, t0_to_test2, target_ID, save_path, '{} folded by {} days'.format(target_ID,period_to_test2))
#        phase_fold_plot(t_cut, BLS_flux, period_to_test3, t0_to_test3, target_ID, save_path, '{} folded by {} days'.format(target_ID,period_to_test3))
#        phase_fold_plot(t_cut, BLS_flux, period_to_test4, t0_to_test4, target_ID, save_path, '{} folded by {} days'.format(target_ID,period_to_test4))
        #print("Absolute amplitude of main variability = {}".format(amplitude_peaks))
        #print('Main Variability Period from Lomb-Scargle = {:.3f}d'.format(p_rot))
        #print("Main Variability Period from BLS of original = {}".format(rot_period))
        #variability_table.add_row([target_ID,p_rot,rot_period,amplitude_peaks])
        
        ############################# Eyeballing ##############################
        """
        Generate 2 x 2 eyeballing plot
        """
        eye_balling_fig, axs = plt.subplots(2,2, figsize = (16,10),  dpi = 120)

        # Original DIA with injected transits setup
        axs[0,0].scatter(lc_30min.time, combined_flux, s=1, c= 'k')
        axs[0,0].set_ylabel('Normalized Flux')
        axs[0,0].set_xlabel('Time')
        axs[0,0].set_title('{} - original {} light curve'.format(target_ID, pipeline))
        #for n in range(int(-1*8/params.per),int(2*8/params.per+2)):
        #    axs[0,0].axvline(params.t0+n*params.per+mid_point, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        
        # Detrended figure setup
        axs[0,1].scatter(t_cut, BLS_flux, c = 'k', s = 1, label = '{} residuals after {} detrending'.format(target_ID,detrending))
        axs[0,1].set_title('{} residuals after {} detrending - Sector {}'.format(target_ID, detrending, sector))
        axs[0,1].set_ylabel('Normalized Flux')
        axs[0,1].set_xlabel('Time - 2457000 [BTJD days]')
        #for n in range(int(-1*8/params.per),int(2*8/params.per+2)):
        #    axs[0,1].axvline(params.t0+n*params.per+mid_point, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        
        # Periodogram setup
        axs[1,0].plot(results.period, results.power, "k", lw=0.5)
        axs[1,0].set_xlim(results.period.min().value, results.period.max().value)
        axs[1,0].set_xlabel("period [days]")
        axs[1,0].set_ylabel("log likelihood")
        axs[1,0].set_title('{} - BLS Periodogram of residuals'.format(target_ID))
        axs[1,0].axvline(period.value, alpha=0.4, lw=3)
        for n in range(2, 10):
            axs[1,0].axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
            axs[1,0].axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
        
        # Folded or zoomed plot setup
#        epoch = t0.value
#        period = period.value
        epoch = 1332.30997
        period = 8.138268
        phase = np.mod(t_cut-epoch-period/2,period)/period 
        axs[1,1].scatter(phase, BLS_flux, c='k', s=1)
        axs[1,1].set_title('{} Lightcurve folded by {:0.4} days'.format(target_ID, period))
        axs[1,1].set_xlabel('Phase')
        axs[1,1].set_ylabel('Normalized Flux')
        
        eye_balling_fig.tight_layout()
#        eye_balling_fig.savefig(save_path + '{} - Full eyeballing fig.png'.format(target_ID))
        #plt.close(eye_balling_fig)
        plt.show(block = False)
        
    except RuntimeError:
        print('No DiffImage lc exists for {}'.format(target_ID))
    #except:
        #print('Some other error for {}'.format(target_ID))

#ascii.write(variability_table, save_path + 'Variability_info_eleanor.csv', format='csv', overwrite = True)        
        

