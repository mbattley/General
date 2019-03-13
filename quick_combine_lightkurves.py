#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:23:33 2019

@author: mbattley
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
from photutils import CircularAperture
from astropy.stats import SigmaClip

tpf1 = lightkurve.search.open('TESS_Sector_1_cutouts/tess-s0001-3-1_20.79808333_-69.36066667_11x11_astrocut.fits')
tpf2 = lightkurve.search.open('TESS_Sector_1_cutouts/tess-s0002-3-3_20.79808333_-69.36066667_11x11_astrocut.fits')

median_image1 = np.nanmedian(tpf1.flux, axis=0)
median_image2 = np.nanmedian(tpf2.flux, axis=0)

# Select pixels which are brighter than the 85th percentile of the median image
#aperture_mask1 = median_image1 > np.nanpercentile(median_image1, 85)
#aperture_mask2 = median_image2 > np.nanpercentile(median_image2, 85)

# Make center aperture for crowded fields
crowded_field_aperture = np.zeros((11,11))
crowded_field_aperture2 = np.zeros((11,11))
crowded_field_aperture[5:8,4:7] = 1
crowded_field_aperture2[4:7,5:8] = 1
crowded_field_aperture = crowded_field_aperture.astype(np.bool)
crowded_field_aperture2 = crowded_field_aperture2.astype(np.bool)
aperture_mask1 = crowded_field_aperture
aperture_mask2 = crowded_field_aperture2


tpf1.plot(aperture_mask = aperture_mask1)
tpf2.plot(aperture_mask = aperture_mask2)

# Plot separate lightcurves
tpf1.to_lightcurve(aperture_mask = aperture_mask1).plot()
tpf2.to_lightcurve(aperture_mask = aperture_mask2).plot()

# Convert to lightcurves
lc1 = tpf1.to_lightcurve(aperture_mask = aperture_mask1)
lc2 = tpf2.to_lightcurve(aperture_mask = aperture_mask2)

end_lc1 = lc1.time[-1]
start_lc2 = lc2.time[0]

print('End of lc1 = {}'.format(end_lc1))
print('Start of lc1 = {}'.format(start_lc2))

# Combine and plot lightcurves
combined_lc = lc1
combined_lc = combined_lc.append(lc2)
combined_lc.plot()

# Remove outliers
#sigma_clipped_lc = combined_lc.remove_outliers(sigma = 3)
#sigma_clipped_lc.plot()

# Perform Dave's flattening on combined lightcurve
time_from_zero = combined_lc.time - combined_lc.time[0]
lcurve = np.vstack((time_from_zero, combined_lc.flux, combined_lc.flux_err)).T
TESSflatten_lc = TESSflatten(lcurve, winsize = 3.5, stepsize = 0.15, gapthresh = 0.1)

# Plot result
plt.figure()
plt.scatter(combined_lc.time[0:2477], TESSflatten_lc, c = 'k', s = 1, label = 'TESSflatten flux')
plt.title('2MASS J01231125-6921379 with TESSflatten - Sectors 1&2')
plt.ylabel('Normalized Flux')
plt.xlabel('Time - 2457000 [BTJD days]')