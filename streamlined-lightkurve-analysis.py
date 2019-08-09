#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:17:35 2019

Streamlined function to plot and save TESS cutouts from filenames

@author: mbattley
"""

import lightkurve
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.stats import BoxLeastSquares
from astropy.coordinates import SkyCoord
from glob import glob
from lightkurve import KeplerTargetPixelFile, TessTargetPixelFile
from TESSselfflatten import TESSflatten
from astroquery.mast import Tesscut
from photutils import MMMBackground, MeanBackground, MedianBackground, ModeEstimatorBackground, SExtractorBackground
from astropy.stats import SigmaClip
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask
from bls import BLS


def make_transit_periodogram(t, y, target_ID, save_path, sector, dy=0.01):
    """
    Plots a periodogram to determine likely period of planet transit candidtaes 
    in a dataset, based on a box least squared method.
    """
    model = BoxLeastSquares(t * u.day, y, dy = 0.01)
    periodogram = model.autopower(0.2, objective="snr")
    periodogram_fig = plt.figure()
    plt.plot(periodogram.period, periodogram.power,'k')
    plt.xlabel('Period [days]')
    plt.ylabel('Power')
    plt.title('Periodogram for {}'.format(target_ID))
    plt.savefig(save_path + '{} - Sector {} - Periodogram.png'.format(target_ID, sector))
    plt.close(periodogram_fig)
    max_power_i = np.argmax(periodogram.power)
    best_fit = periodogram.period[max_power_i]
    print('Best Fit Period: {} days'.format(best_fit))
    stats = model.compute_stats(periodogram.period[max_power_i],
                                periodogram.duration[max_power_i],
                                periodogram.transit_time[max_power_i])
    return stats, best_fit
    
    # Find optimum period (outdated)
    #periods = np.arange(0.3, 8, 0.0001)
    #durations = np.arange(0.005, 0.15, 0.001)
    #power, _, _, _, _, _, _ = transit_periodogram(time=lc.time,
    #                                              flux=lc.flux,
    #                                              flux_err=lc.flux_err,
    #                                              periods=periods,
    #                                              durations=durations)
    #best_fit = periods[np.argmax(power)]
    #print('Best Fit Period: {} days'.format(best_fit))

def bls_search(lc, target_ID, save_path):
    """
    Perform bls analysis using foreman-mackey's bls.py function
    """
    durations = np.linspace(0.05, 0.2, 22) * u.day
    model = BLS(lc.time*u.day, lc.flux)
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
    #plt.savefig(save_path + '{} - BLS Periodogram.png'.format(target_ID))
#    plt.close(fig)
    
    
    # Fold by most significant period
    phase_fold_plot(lc.time*u.day, lc.flux, period, epoch, target_ID, save_path, title='{} Lightcurve folded by {} days'.format(target_ID, period))
    
    return results, transit_info
    

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
    plt.savefig(save_path + '{} - Phase fold plot.png'.format(target_ID))
#    plt.close(phase_fold_fig)
    
def transit_dot_times(epoch, p_period, lc_time):
    """
    Determines the times for the positions of transits for known exoplanets in
    a lightcurve, given an epoch of first trasnit and period of planet
    """
    p_times = [epoch]
    time = epoch + p_period
    counter = 1
    
    while time < lc_time[-1]:
        p_times += [epoch + counter*p_period]
        counter += 1
        time += p_period
    
    return p_times

def bkg_subtraction(time, flux, scope="tpf", sigma=3):
    """Subtracts background flux from target pixel file.

    Parameters
    ----------
    scope : string, "tpf" or "postcard"
        If `tpf`, will use data from the target pixel file only to estimate and remove the background.
        If `postcard`, will use data from the entire postcard region to estimate and remove the background.
    sigma : float
        The standard deviation cut used to determine which pixels are representative of the background in each cadence.
    """

    tpf_flux_bkg = []

    sigma_clip = SigmaClip(sigma=sigma)
#    bkg = MMMBackground(sigma_clip=sigma_clip)
    bkg = SExtractorBackground(sigma_clip=sigma_clip)

    for i in range(len(time)):
        bkg_value = bkg.calc_background(flux[i])
        tpf_flux_bkg.append(bkg_value)

    tpf_flux_bkg = np.array(tpf_flux_bkg)
    
    return tpf_flux_bkg

def lightkurve_analysis(filename, target_ID, save_path):
    """
    Complete lightkurve analysis similar to that in 'lightkurve_practice.py' 
    but in function form, so can be called for every different star in turn
    """
    
    # Load tpf
    tpf = lightkurve.search.open(filename)
    #tpf = lightkurve.search.search_targetpixelfile(target_ID, sector = 2)
    
    # Attach target name to tpf
    tpf.targetid = target_ID
    
    # Create a median image of the source over time
    median_image = np.nanmedian(tpf.flux, axis=0)
    
    # Select pixels which are brighter than the 85th percentile of the median image
    aperture_mask = median_image > np.nanpercentile(median_image, 85)
    
    # Get flux background and subtract
    #tpf_flux_bkg = bkg_subtraction(tpf.time, tpf.flux)
    #tpf.flux = tpf.flux - tpf_flux_bkg

    # Plot and save tpf
    tpf_plot = tpf.plot(aperture_mask = aperture_mask).get_figure()
    tpf_plot.savefig(save_path + '{} - Sector {} - tpf plot.png'.format(target_ID, tpf.sector))
#    plt.close(tpf_plot)
    
    # Remove outliers and save
    sigma_cut_lc_fig = tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers(sigma = 3).plot().get_figure()
    sigma_cut_lc_fig.savefig(save_path + '{} - Sector {} - 3 sigma lightcurve.png'.format(target_ID, tpf.sector))
#    plt.close(sigma_cut_lc_fig)
    
    # Convert to lightcurve object
    lc = tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers(sigma = 3)
    
    # Set up lightcurve for Dave's flattening code ((nx3) array; time starts from zero)
    time_from_zero = lc.time - lc.time[0]
    lcurve = np.vstack((time_from_zero, lc.flux, lc.flux_err)).T
    
    # Adding in dots under each transit
    # Sector 1 Epochs
    #epoch_p1 = 1385.774883
    #epoch_p2 = 1384.085367
    
    #epoch_p1 = 1439.135
    #period_p1 = 1.0817
    
    # Sector 2 epochs
    #epoch_p1 = 1415.634
    #epoch_p2 = 1412.774
    #period_p1 = 5.97188
    #period_p2 = 3.586159
    
    #p1_times = transit_dot_times(epoch_p1, period_p1, lc.time)
    #p2_times = transit_dot_times(epoch_p2, period_p2, lc.time)
    
    #p1_marker_y = [0.999]*len(p1_times)
    #p2_marker_y = [0.999]*len(p2_times)
    
    ## Run Dave's flattening code
    TESSflatten_fig = plt.figure()
    TESSflatten_lc = TESSflatten(lcurve, winsize = 3.5, stepsize = 0.15, gapthresh = 0.1)
    plt.scatter(lc.time, TESSflatten_lc, c = 'k', s = 1, label = 'TESSflatten flux')
    #plt.scatter(p1_times, p1_marker_y, c = 'r', s = 5, label = 'Planet 1')
    #plt.scatter(p2_times, p2_marker_y, c = 'g', s = 5, label = 'Planet 2')
    plt.title('{} with TESSflatten - Sector {}'.format(tpf.targetid, tpf.sector))
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time - 2457000 [BTJD days]')
    plt.savefig(save_path + '{} - Sector {} - TESSflatten lightcurve.png'.format(target_ID, tpf.sector))
#    plt.close(TESSflatten_fig)
    
    lc.flux = TESSflatten_lc
    
    # Phase folding by periods of suspected planets
    #phase_fold_plot(lc.time, TESSflatten_lc, period_p1, epoch_p1, title='TOI-440 Lightcurve folded by {} days - Sector 5'.format(period_p1))
    #phase_fold_plot(lc.time, TESSflatten_lc, period_p2, epoch_p2, title='TOI-396 Lightcurve folded by {} days - Sector 3'.format(period_p2))

    # Find optimum period
    #stats, best_fit_period = make_transit_periodogram(t = lc.time, y = TESSflatten_lc, target_ID = target_ID, save_path = save_path, sector = tpf.sector)
    
    #Perform BLS search
    bls_search(lc, target_ID, save_path)
    
    # Re-plot
    #lc.plot()
    
    # Phase fold
    #lc.fold(period=4.85374).errorbar() #n.b. period in days
    
    return lc
 
start = time.time()    

#save_path = '/Users/mbattley/Documents/PhD/Lightkurve/YSO-BANYAN-targets/Sector 1/' # laptop
save_path = '/home/astro/phrhzn/Documents/PhD/Promising Star Followup/' # Desktop
#target_ID = 'HD 207043'
#filename = 'TESS_Sector_1_cutouts/tess-s0001-2-4_326.981079166667_-52.9310083333333_11x11_astrocut.fits'

# Alternatively: Get tpf from TESS FFI cutouts
#cutout_coord = SkyCoord(42.984364, -30.814529, unit="deg")
#manifest = Tesscut.download_cutouts(cutout_coord, [10,15])
#print(manifest)

with open('Sector_1_targets.pkl', 'rb') as f:
    target_list = pickle.load(f)
f.close()
with open('Sector_1_target_filenames.pkl', 'rb') as f:
    target_filenames = pickle.load(f)
f.close()

target_list = ['CD-60 416']

for target_ID in target_list:
    filenames = target_filenames[target_ID]
#    filenames = 'hi'
    if len(filenames) == 0:
        print('{} has no associated files'.format(target_ID))
    elif type(filenames) == str:
        lightkurve_analysis(filenames, target_ID, save_path)
    else:
        combined_lc = lightkurve_analysis(filenames[0], target_ID, save_path)
        for filename in filenames[1:]:
            sector_lc = lightkurve_analysis(filename, target_ID, save_path)
            combined_lc = combined_lc.append(sector_lc)
        combined_lc_fig = plt.figure()
        plt.scatter(combined_lc.time, combined_lc.flux, c = 'k', s = 1)
        #plt.scatter(p1_times, p1_marker_y, c = 'r', s = 5, label = 'Planet 1')
        #plt.scatter(p2_times, p2_marker_y, c = 'g', s = 5, label = 'Planet 2')
        plt.title('Combined lightcurve for {}'.format(target_ID))
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time - 2457000 [BTJD days]')
        plt.savefig(save_path + '{} - Combined_lightcurve.png'.format(target_ID))
#        plt.close(combined_lc_fig)
        #stats, best_fit_period = make_transit_periodogram(t = combined_lc.time, y = combined_lc.flux, target_ID = target_ID, save_path = save_path, sector = 'Multiple')
        bls_search(combined_lc, target_ID, save_path)
        
end = time.time()
print(end - start)