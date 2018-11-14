#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:52:44 2018

Examination of the area around the Sco-OB2 Stellar association

This script examines the area around the 'known' members of the Sco-OB2 Stellar
Association, and plots a series of plots in order to take steps towards identifying 
new members of this association.

Inputs: A VOTable containing Gaia DR2 data for the area around the Sco-OB2 
        Stellar Association, including position, velocity, magnitude and indetifying
        information.

Ouputs: Proper motion density plot
        Full area CAMD diagram
        Plot of stellar positions in this area

@author: Matthew Battley
"""

import astropy.table as tab
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import timeit

start = timeit.default_timer()

def plot_with_colourbar(x,y,mag,xlabel,ylabel,title,invert_y_axis = False):
    """
    Function for plotting a scatter plot in two variables (x,y) with a colour bar based on a third (mag)
    """
    # Sets up colours and normalisation for colourbar
    cmap = matplotlib.cm.get_cmap('rainbow')
    normalize = matplotlib.colors.Normalize(vmin = min(mag), vmax=max(mag))
    colours = [cmap(normalize(value)) for value in mag]
    
    # Plots figure
    fig_pos, ax = plt.subplots(figsize=(10,10))
    plt.scatter(x,y,0.5,c=colours)
    if invert_y_axis == True:
        plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm = normalize)
    cbar.set_label('g Magnitude')

# Read data from table
Table = tab.Table
data = Table.read('OB2_All_boxes_bp-rp')
hipparcos_data = Table.read('Hipparcos_OB2_de_Zeeuw_1999.vot')

# Change from unrecognisable unit names in file
data['pmra'].unit = 'mas/yr'
data['pmdec'].unit = 'mas/yr'
data['radial_velocity'].unit = 'km/s'
hipparcos_data['pmra'].unit = 'mas/yr'
hipparcos_data['pmdec'].unit = 'mas/yr'
hipparcos_data['ra'].unit = 'deg'
hipparcos_data['dec'].unit = 'deg'

# Input sky coordinates for all stars
c_icrs = SkyCoord(ra = data['ra'], dec = data['dec'], pm_ra_cosdec = data['pmra'], pm_dec = data['pmdec'])
c_icrs_hipparcos = SkyCoord(ra = hipparcos_data['ra'], dec = hipparcos_data['dec'], pm_ra_cosdec = hipparcos_data['pmra'], pm_dec = hipparcos_data['pmdec'])
print(c_icrs)

# Convert star coordinates to Galactic frame
c_galactic = c_icrs.galactic
c_galactic_hipparcos = c_icrs_hipparcos.galactic
print(c_galactic)

# Add equivalent galactic coordinates back into data
data['pm_l_cosb'] = c_galactic.pm_l_cosb
data['pm_b'] = c_galactic.pm_b
hipparcos_data['pm_l_cosb'] = c_galactic_hipparcos.pm_l_cosb
hipparcos_data['pm_b'] = c_galactic_hipparcos.pm_b

# Select stars within this data where pms are only in the region between pm_l = [-50,10] and pm_b = [-30,30]
sel = data['pm_l_cosb'] >= -50
sel &= data['pm_l_cosb'] < 10
sel &= data['pm_b'] >= -30
sel &= data['pm_b'] <= 30

small_area_stars = data[sel]
print(len(small_area_stars['pmra']))

# Plotting proper motion density plot
from scipy.stats import kde
plt.figure()
k = kde.gaussian_kde([small_area_stars['pm_l_cosb'], small_area_stars['pm_b']])
nbins = 100
xi, yi = np.mgrid[small_area_stars['pm_l_cosb'].min():small_area_stars['pm_l_cosb'].max():nbins*1j, small_area_stars['pm_b'].min():small_area_stars['pm_b'].max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r, vmax = 0.002)
plt.colorbar()
plt.xlabel('pm_l_cosb (mas/yr)')
plt.ylabel('pm_b (mas/yr)')
plt.title('Proper motion plot for potential Sco_OB2 members')
plt.scatter(hipparcos_data['pm_l_cosb'],hipparcos_data['pm_b'], 0.1, 'k')

# Plots star positions
plot_with_colourbar(data['ra'],data['dec'],data['phot_g_mean_mag'],'ra (mas)','dec (mas)','Location plot - OB2 - All boxes in TESS Mag range')

# Removes empty data from bp-rp
data['bp_rp'].filled(-999)
real_indices = [i for i, x in enumerate(data['bp_rp']) if x != -999]
bp_rp = data['bp_rp'][real_indices] # Removes empty data from bp-rp
mag_4_CAMD = data['phot_g_mean_mag'][real_indices]

# Plots Colour-Absolute Magnitude Diagram
plot_with_colourbar(bp_rp,mag_4_CAMD,mag_4_CAMD,'BP-RP','Gaia G-band Magnitude','Colour-Absolute Magnitude Diagram for Stellar Association OB2 - All Boxes in TESS Mag range',invert_y_axis = True)

stop = timeit.default_timer()

print('Time: ',stop - start)