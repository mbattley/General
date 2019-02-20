#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:26:48 2019

Transitleastsquares practice

@author: phrhzn
"""

import lightkurve
import matplotlib.pyplot as plt
import numpy as np
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask
#
######################### Import and tidy data  ################################
tpf = lightkurve.search_targetpixelfile("TIC 238196350", sector=1).download()

aperture_mask = tpf.pipeline_mask

# Plot tpf
tpf.plot(aperture_mask = tpf.pipeline_mask)

# Create a median image of the source over time
median_image = np.nanmedian(tpf.flux, axis=0)

# Select pixels which are brighter than the 85th percentile of the median image
aperture_mask = median_image > np.nanpercentile(median_image, 85)

# Plot tpf
tpf.plot(aperture_mask = aperture_mask)
#
# Plot base lightcurve
#tpf.to_lightcurve().plot()
tpf.to_lightcurve(aperture_mask = aperture_mask).plot()

# Flatten lightcurve
tpf.to_lightcurve(aperture_mask = aperture_mask).flatten().plot()

# Remove outliers
tpf.to_lightcurve(aperture_mask = aperture_mask).flatten(window_length = 1001).remove_outliers().plot()
#plt.xlim(1330, 1335)

# Now with binning!
tpf.to_lightcurve(aperture_mask = aperture_mask).flatten(window_length = 1001).remove_outliers().bin(binsize=10).plot()
#plt.xlim(1330, 1335)

# Convert to lightcurve
lc = tpf.to_lightcurve(aperture_mask = aperture_mask).flatten(window_length=1001).remove_outliers()

# Clip out dodgy jitter data
lc = lc[(lc.time < 1346) | (lc.time > 1350)]
#
## Bin data if desired
#lc = lc.bin(binsize = 10)

###################### transitleastsquares modelling ###########################

#Perform TransitLeastSquares transit search
model = transitleastsquares(lc.time, lc.flux)
results = model.power(oversampling_factor=5, duration_grid_step=1.02)

#Plot power spectrum and integer (sub)harmonics
plt.figure()
ax = plt.gca()
ax.axvline(results.period, alpha=0.4, lw=3)
plt.xlim(np.min(results.periods), np.max(results.periods))
for n in range(2, 10):
    ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
    ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
plt.ylabel(r'SDE')
plt.xlabel('Period (days)')
plt.plot(results.periods, results.power, color='black', lw=0.5)
plt.xlim(0, max(results.periods))

# Inspect statistics
print('Period', format(results.period, '.5f'), 'd')
print(len(results.transit_times), 'transit times in time series:', \
        ['{0:0.5f}'.format(i) for i in results.transit_times])
print('Transit depth', format(results.depth, '.5f'))
print('Transit duration (days)', format(results.duration, '.5f'))

# Visualise fit on phase-folded curve
plt.figure()
plt.plot(
    results.model_folded_phase,
    results.model_folded_model,
    color='red')
plt.scatter(
    results.folded_phase,
    results.folded_y,
    color='blue',
    s=10,
    alpha=0.5,
    zorder=2)
#plt.xlim(0.49, 0.51)
plt.xlabel('Phase')
plt.ylabel('Relative flux')

# Plot over original data
plt.figure()
in_transit = transit_mask(
    lc.time,
    results.period,
    results.duration,
    results.T0)
plt.scatter(
    lc.time[in_transit],
    lc.flux[in_transit],
    color='red',
    s=2,
    zorder=0)
plt.scatter(
    lc.time[~in_transit],
    lc.flux[~in_transit],
    color='blue',
    alpha=0.5,
    s=2,
    zorder=0)
plt.plot(
    results.model_lightcurve_time,
    results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
plt.xlim(min(lc.time), max(lc.time))
#plt.ylim(0.9985, 1.0003)
plt.xlabel('Time (days)')
plt.ylabel('Relative flux')