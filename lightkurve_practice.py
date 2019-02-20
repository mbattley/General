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
from transit_periodogram import transit_periodogram
import astropy.units as u
from astropy.stats import BoxLeastSquares
from astropy.coordinates import SkyCoord
from glob import glob
from lightkurve import KeplerTargetPixelFile, TessTargetPixelFile
from TESSselfflatten import TESSflatten

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
    return stats
    
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

def phase_fold_plot(t, lc, period, epoch, title):
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

# Get target pixel file
#tpf = lightkurve.search_targetpixelfile('kepler-10', quarter=5).download()
#tpf = lightkurve.search_targetpixelfile("316.9615, -26.0967", sector=1).download()
#tpf = lightkurve.search_targetpixelfile("TIC 31747041", sector=1).download()
#tpf = lightkurve.search_targetpixelfile("WASP 73", sector=1).download()
#tpf = lightkurve.search_targetpixelfile("21:06:31.65 -26:41:34.29 ", sector=1).download()
#tpf = lightkurve.search.open('tess-s0001-1-4_316.63187500000004_-26.692858333333334_10x10_astrocut.fits')
#tpf = TessTargetPixelFile('https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00261136679-s01_tess_v1_tp.fits')
#tpf = lightkurve.search.open('tess-s0003-2-2_42.984364_-30.814529_10x15_astrocut.fits')
tpf = lightkurve.search.open('tess-s0004-2-1_42.984364_-30.814529_10x15_astrocut.fits')
#lc = lightkurve.search_tesscut(" TIC 178155732", sector =3)
#
## Alternatively: Get tpf from TESS FFI cutouts
##fnames = np.sort(glob('*.fits'))
##tpf = KeplerTargetPixelFile.from_fits_images(images=fnames,
##                                             position=SkyCoord(84.2875, -80.46889, unit='deg'),
##                                             size=(10,10),
##                                             target_id='Pi_Men_c')
##tpf = open("tess-s0001-1-4_316.631875_-26.6928583_10x10_astrocut.fits")
##
#aperture_mask = tpf.pipeline_mask

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
#
# Plot tpf
tpf.plot(aperture_mask = aperture_mask)
#
# Plot base lightcurve
#tpf.to_lightcurve().plot()
tpf.to_lightcurve(aperture_mask = aperture_mask).plot()

# Flatten lightcurve
#tpf.to_lightcurve(aperture_mask = aperture_mask).flatten().plot()

# Remove outliers
#tpf.to_lightcurve(aperture_mask = aperture_mask).flatten(window_length = 1001).remove_outliers().plot()
tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers(sigma = 5).plot()
#plt.xlim(1330, 1335)

# Now with binning!
#tpf.to_lightcurve(aperture_mask = aperture_mask).flatten(window_length = 1001).remove_outliers().bin(binsize=10).plot()
#plt.xlim(1330, 1335)

# Convert to lightcurve
lc = tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers(sigma = 5)

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

# Sector 2 epochs
epoch_p1 = 1415.634
epoch_p2 = 1412.774
period_p1 = 5.97188
period_p2 = 3.586159

p1_times = transit_dot_times(epoch_p1, period_p1, lc.time)
p2_times = transit_dot_times(epoch_p2, period_p2, lc.time)

p1_marker_y = [0.999]*len(p1_times)
p2_marker_y = [0.999]*len(p2_times)

## Run Dave's flattening code
plt.figure()
TESSflatten_lc = TESSflatten(lcurve, gapthresh = 1)
plt.scatter(lc.time, TESSflatten_lc, c = 'k', s = 1, label = 'TESSflatten flux')
plt.scatter(p1_times, p1_marker_y, c = 'r', s = 5, label = 'Planet 1')
plt.scatter(p2_times, p2_marker_y, c = 'g', s = 5, label = 'Planet 2')
plt.title('TOI-396 with TESSflatten - Sector 4')
plt.ylabel('Normalized Flux')
plt.xlabel('Time - 2457000 [BTJD days]')

# Phase folding by periods of suspected planets
phase_fold_plot(lc.time, TESSflatten_lc, period_p1, epoch_p1, title='TOI-396 Lightcurve folded by {} days - Sector 4'.format(period_p1))
phase_fold_plot(lc.time, TESSflatten_lc, period_p2, epoch_p2, title='TOI-396 Lightcurve folded by {} days - Sector 4'.format(period_p2))

#
## Clip out dodgy jitter data
#lc = lc[(lc.time < 1382) | (lc.time > 1384)]
#
## Find optimum period
#make_transit_periodogram(t = lc.time, y = lc.flux)
#
## Re-plot
#lc.plot()

# Phase fold
#lc.fold(period=4.85374).errorbar() #n.b. period in days

##Phase fold without flattening
#tpf.to_lightcurve().fold(period=2.849375).errorbar()
#tpf.to_lightcurve(aperture_mask = aperture_mask).remove_outliers().fold(period=3.0440499).errorbar()

##############################################################################

# And now for Pi Men c:
#tpf2 = TessTargetPixelFile('https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00261136679-s01_tess_v1_tp.fits')
#
## Create better aperture...
## Create a median image of the source over time
#median_image = np.nanmedian(tpf2.flux, axis=0)
#
## Select pixels which are brighter than the 85th percentile of the median image
#aperture_mask = median_image > np.nanpercentile(median_image, 85)
#
## Plot that aperture
#tpf2.plot(aperture_mask=aperture_mask)
#
## Convert to lightcurve
#lc2 = tpf2.to_lightcurve(aperture_mask=aperture_mask).flatten(window_length=1001)
#
## Clip out dodgy jitter data
#lc2 = lc2[(lc2.time < 1346) | (lc2.time > 1350)]
#
#
## Find optimum period (new)
#make_transit_periodogram(t = lc2.time, y = lc2.flux)
#
##Remove outliers, fold, bin and plot with error bars
#lc2.remove_outliers(sigma=6).fold(period=6.275).bin(binsize=10).errorbar()

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