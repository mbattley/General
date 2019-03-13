#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:19:18 2019

A collecction of useful lightkurve calls and general practice

@author: phrhzn
"""

import lightkurve
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
from photutils import CircularAperture, RectangularAperture
from astropy.stats import SigmaClip
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask
from bls import BLS
from matplotlib import rcParams


def make_transit_periodogram(t,y,dy=0.01):
    """
    Plots a periodogram to determine likely period of planet transit candidtaes 
    in a dataset, based on a box least squared method.
    """
    model = BoxLeastSquares(t * u.day, y, dy = 0.01)
    periodogram = model.autopower(0.2, objective="snr")
    plt.figure()
    plt.plot(periodogram.period, periodogram.power,'k')
    plt.xlabel('Period [days]')
    plt.ylabel('Power')
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

def phase_fold_plot(t, lc, period, epoch, title, save_path = '/Users/mbattley/Documents/PhD/Lightkurve/YSO-BANYAN-targets/Sector 1/'):
    """
    Phase-folds the lc by the given period, and plots a phase-folded light-curve
    for the object of interest
    """
    phase = np.mod(t-epoch-period/2,period)/period 
    
    plt.figure()
    plt.scatter(phase, lc, c='k', s=2)
    plt.title(title)
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.savefig(save_path + '{} - Phase folded lightcurve.png'.format(target_ID))
    
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
    
    bkg_MMM = MMMBackground(sigma_clip=sigma_clip)
    bkg_ModeEstimator = ModeEstimatorBackground(median_factor=3., mean_factor=2.,sigma_clip=sigma_clip)
    bkg_Mean = MeanBackground(sigma_clip)
    bkg_Median = MedianBackground(sigma_clip)
    bkg_SExtractor = SExtractorBackground(sigma_clip)
    
    bkg_MMM_value = bkg_MMM.calc_background(flux[0])
    bkg_ModeEstimator_value = bkg_ModeEstimator.calc_background(flux[0])
    bkg_Mean_value = bkg_Mean.calc_background(flux[0])
    bkg_Median_value = bkg_Median.calc_background(flux[0])
    bkg_SExtractor_value = bkg_SExtractor.calc_background(flux[0])
    
    print("MMM Background = {}".format(bkg_MMM_value))
    print("ModeEstimator Background = {}".format(bkg_ModeEstimator_value))
    print("Mean Background = {}".format(bkg_Mean_value))
    print("Median Background = {}".format(bkg_Median_value))
    print("SExtractor Background = {}".format(bkg_SExtractor_value))

    for i in range(len(time)):
        bkg_value = bkg.calc_background(flux[i])
        tpf_flux_bkg.append(bkg_value)

    tpf_flux_bkg = np.array(tpf_flux_bkg)
    
    return tpf_flux_bkg

# Get target pixel file
    
target_ID = 'TOI 396'
#save_path = '/home/astro/phrhzn/Documents/PhD/Lightkurve/YSO-BANYAN-targets/Sector 1/' # On Desktop
save_path = '/Users/mbattley/Documents/PhD/Lightkurve/YSO-BANYAN-targets/Sector 1/' #On laptop

#tpf = lightkurve.search_targetpixelfile('kepler-10', quarter=5).download()
#tpf = lightkurve.search_targetpixelfile("316.9615, -26.0967", sector=1).download()
#tpf = lightkurve.search_targetpixelfile("TIC 31747041", sector=1).download()
#tpf = lightkurve.search_targetpixelfile("WASP 73", sector=1).download()
#tpf = lightkurve.search_targetpixelfile("21:06:31.65 -26:41:34.29 ", sector=1).download()
#tpf = lightkurve.search.open('tess-s0001-1-4_316.63187500000004_-26.692858333333334_10x10_astrocut.fits')
#tpf = TessTargetPixelFile('https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00261136679-s01_tess_v1_tp.fits')
tpf = lightkurve.search.open('tess-s0003-2-2_42.984364_-30.814529_10x15_astrocut.fits') #- TOI 396
#tpf = lightkurve.search.open('tess-s0004-2-1_42.984364_-30.814529_10x15_astrocut.fits')
#tpf = lightkurve.search.open('tess-s0001-2-4_319.9496107691067_-58.1488869525922_11x11_astrocut.fits') # Wasp 73
#tpf = lightkurve.search.open('tess-s0001-2-4_326.980375_-52.9306389_11x11_astrocut.fits')
#lc = lightkurve.search_tesscut(" TIC 178155732", sector =3)
#
# Alternatively: Get tpf from TESS FFI cutouts
#cutout_coord = SkyCoord(42.984364, -30.814529, unit="deg")
#manifest = Tesscut.download_cutouts(cutout_coord, [10,15])
#print(manifest)

#aperture_mask = tpf.pipeline_mask

# Attach target name to tpf
tpf.targetid = target_ID

# Plot tpf
tpf.plot(aperture_mask = tpf.pipeline_mask)

# Create a median image of the source over time
median_image = np.nanmedian(tpf.flux, axis=0)

# Select pixels which are brighter than the 85th percentile of the median image
aperture_mask = median_image > np.nanpercentile(median_image, 85)

##Create my own mask (if necessary)
##my_mask= np.array([[False, False, False, False, False, False, False, False, False,
##        False],
##       [False, False, False, False, False, False, True, True, True,
##        False],
##       [False, False, False, False, False, False, True, True, True,
##        False],
##       [False, False, False, False, False, False, True, True, True,
##        False],
##       [False, False, False, False, False, False, False, False, False,
##        False],
##       [False, False, False, False, False,  False,  False,  False, False,
##        False],
##       [False, False, False, False,  False,  False,  False,  False,  False,
##        False],
##       [False, False, False, False,  False,  False,  False,  False,  False,
##        False],
##       [False, False, False, False,  False,  False,  False,  False,  False,
##        False],
##       [False, False, False, False,  False,  False,  False,  False,  False,
##        False]])
#

#my_mask = np.array([[False, False, False, False, False],
#                    [False, False, True, True, False],
#                    [False, False, True, True, False],
#                    [False, False, True, True, False],
#                    [False, False, False, False, False]])

#aperture_mask = my_mask
    
tpf_flux_bkg = bkg_subtraction(tpf.time, tpf.flux)

# Plot tpf
tpf_plot = tpf.plot(aperture_mask = aperture_mask).get_figure()
#tpf_plot.savefig(save_path + '{} - tpf plot.png'.format(target_ID))
#plt.close(tpf_plot)

# Plot base lightcurve
#tpf.to_lightcurve().plot()
tpf.to_lightcurve(aperture_mask = aperture_mask).plot()

# Flatten lightcurve
#tpf.to_lightcurve(aperture_mask = aperture_mask).flatten().plot()

# Remove outliers and save
#tpf.to_lightcurve(aperture_mask = aperture_mask).flatten(window_length = 1001).remove_outliers().plot()
sigma_cut_lc_fig = tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers(sigma = 3).plot().get_figure()
#sigma_cut_lc_fig.savefig(save_path + '{} - 3 sigma lightcurve.png'.format(target_ID))
#plt.close(sigma_cut_lc_fig)
#plt.xlim(1330, 1335)

# Bin lightcurve
#tpf.to_lightcurve(aperture_mask = aperture_mask).flatten(window_length = 1001).remove_outliers().bin(binsize=10).plot()
#plt.xlim(1330, 1335)

# Convert to lightcurve object
lc = tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers(sigma = 3)
#lc = tpf.to_lightcurve(aperture_mask = aperture_mask)

## Clip out dodgy jitter data
#lc = lc[(lc.time < 1382) | (lc.time > 1384)]

#lc.flatten().fold(period=5.97188).scatter()

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
#plt.savefig(save_path + '{} - TESSflatten lightcurve.png'.format(target_ID))
#plt.close(TESSflatten_fig)

# Phase folding by periods of suspected planets
#phase_fold_plot(lc.time, TESSflatten_lc, period_p1, epoch_p1, title='TOI-440 Lightcurve folded by {} days - Sector 5'.format(period_p1))
#phase_fold_plot(lc.time, TESSflatten_lc, period_p2, epoch_p2, title='TOI-396 Lightcurve folded by {} days - Sector 3'.format(period_p2))

#
## Clip out dodgy jitter data
#lc = lc[(lc.time < 1382) | (lc.time > 1384)]
#
## Find optimum period
stats, best_fit_period = make_transit_periodogram(t = lc.time, y = TESSflatten_lc)
#
## Re-plot
#lc.plot()

# Phase fold
#lc.fold(period=4.85374).errorbar() #n.b. period in days

##Phase fold without flattening
#tpf.to_lightcurve().fold(period=2.849375).errorbar()
#tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers().fold(period=3.0440499).errorbar()

########################transitleastsquares stuff#############################

#Perform TransitLeastSquares transit search
#model = transitleastsquares(lc.time, TESSflatten_lc)
#results = model.power(oversampling_factor=5, duration_grid_step=1.02)
#
##Plot power spectrum and integer (sub)harmonics
#plt.figure()
#ax = plt.gca()
#ax.axvline(results.period, alpha=0.4, lw=3)
#plt.xlim(np.min(results.periods), np.max(results.periods))
#for n in range(2, 10):
#    ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
#    ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
#plt.ylabel(r'SDE')
#plt.xlabel('Period (days)')
#plt.plot(results.periods, results.power, color='black', lw=0.5)
#plt.xlim(0, max(results.periods))


##############################################################################

# Comparing two apertures

## Use the default
#lc3 = tpf2.to_lightcurve(aperture_mask=tpf.pipeline_mask).flatten(window_length=1001)
#lc3 = lc3[(lc3.time < 1346) | (lc3.time > 1350)].remove_outliers(6).fold(period=6.27, phase=0.4).bin(10)
#
## Use a custom aperture
#custom_lc3 = tpf2.to_lightcurve(aperture_mask=aperture_mask).flatten(window_length=1001)
#custom_lc3 = custom_lc3[(custom_lc3.time < 1346) | (custom_lc3.time > 1350)].remove_outliers(6).fold(period=6.27, phase=0.4).bin(10)
#
## Plot both
#ax = lc3.errorbar(label='Default aperture')
#custom_lc3.errorbar(ax=ax, label='Custom aperture')

############################ bls.py stuff #####################################
durations = np.linspace(0.05, 0.2, 22) * u.day
model = BLS(lc.time*u.day, TESSflatten_lc)
results = model.autopower(durations, frequency_factor=5.0)

# Find the period and epoch of the peak
index = np.argmax(results.power)
period = results.period[index]/2
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
fig.savefig(save_path + '{} - BLS Periodogram.png'.format(target_ID))


# Fold by most significant period
phase_fold_plot(lc.time*u.day, TESSflatten_lc, period, epoch, title='{} Lightcurve folded by {} days'.format(target_ID, period))


lc.flux = TESSflatten_lc
lc.fold(period=3.586).scatter()
lc.fold(period=5.97294).scatter()
lc.fold(period=11.23).scatter()