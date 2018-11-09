#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:22:18 2018

Testing alternative table reading method with astropy
@author: Matthew Battley
"""

import astropy.table as tab
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

# Read data from table
Table = tab.Table
data = Table.read('OB2_All_boxes_bp-rp')

# Change from unrecognisable unit name in file
data['pmra'].unit = 'mas/yr'
data['pmdec'].unit = 'mas/yr'

# Input sky coordinates for all stars
c_icrs = SkyCoord(ra = data['ra'], dec = data['dec'], pm_ra_cosdec = data['pmra'], pm_dec = data['pmdec'])
print(c_icrs)

# Convert star coordinates to Galactic frame
c_galactic = c_icrs.galactic
print(c_galactic)

# Add equivalent galactic coordinates back into data
data['pm_l_cosb'] = c_galactic.pm_l_cosb
data['pm_b'] = c_galactic.pm_b

# Select stars within this data where pms are only in the region between pm_l = [-50,10] and pm_b = [-30,30]
sel = data['pm_l_cosb'] >= -50
sel &= data['pm_l_cosb'] < 10
sel &= data['pm_b'] >= -30
sel &= data['pm_b'] <= 30

small_area_stars = data[sel]
print(len(small_area_stars['pmra']))

# Plotting density plot
from scipy.stats import kde
plt.figure()
k = kde.gaussian_kde([small_area_stars['pm_l_cosb'], small_area_stars['pm_b']])
nbins = 300
xi, yi = np.mgrid[small_area_stars['pm_l_cosb'].min():small_area_stars['pm_l_cosb'].max():nbins*1j, small_area_stars['pm_b'].min():small_area_stars['pm_b'].max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
plt.colorbar()

print('finished')