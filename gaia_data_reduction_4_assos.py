#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:52:44 2018

Gaia Data reduction to find YSOs in stellar associations and clusters 

This script examines the wide area around the 'known' members of a stellar 
association or cluster, before reducing the data to potential young members of 
this group of stars

Input : A VOTable containing Gaia DR2 data for the area around the association 
        of interest, including position, distance, velocity, magnitude and 
        identifying information.

Ouputs: Proper motion density plot for full area 
            - with or without selection polygon
        Full area CAMD diagram
        Plot of stellar positions in this area
        
        Selected area CAMD diagram
        CAMD density plot for selected area
            - with or without selection polygon
        
        Final proper motion and location plots for fully reduced data
        

@author: Matthew Battley
"""

import astropy.table as tab
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import timeit
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.stats import kde

start = timeit.default_timer()

def plot_with_colourbar(x,y,mag,xlabel,ylabel,title,cbar_label = 'g Magnitude' ,invert_y_axis = False, y_lim = False):
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
    if y_lim != False:
        plt.gca().set_ylim(y_lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm = normalize)
    cbar.ax.invert_yaxis()
    cbar.set_label(cbar_label)

######################## IMPORTS AND SORTS OUT DATA ###########################

# Read data from table
Table = tab.Table
data = Table.read('BPMG area 15-24hr_dist.vot')
confirmed_data = Table.read('BPMG_All')

# Change from unrecognisable unit names in file
data['pmra'].unit = 'mas/yr'
data['pmdec'].unit = 'mas/yr'
data['radial_velocity'].unit = 'km/s'
confirmed_data['pmra_x'].unit = 'mas/yr'
confirmed_data['pmdec_x'].unit = 'mas/yr'
confirmed_data['ra'].unit = 'deg'
confirmed_data['dec'].unit = 'deg'

# Input sky coordinates for all stars
c_icrs = SkyCoord(ra = data['ra'], dec = data['dec'], pm_ra_cosdec = data['pmra'], pm_dec = data['pmdec'])
#c_icrs_hipparcos = SkyCoord(ra = confirmed_data['ra'], dec = confirmed_data['dec'], pm_ra_cosdec = confirmed_data['pmra'], pm_dec = confirmed_data['pmdec'])
c_icrs_hipparcos = SkyCoord(ra = confirmed_data['ra'], dec = confirmed_data['dec'], pm_ra_cosdec = confirmed_data['pmra_x'], pm_dec = confirmed_data['pmdec_x'])


# Convert star coordinates to Galactic frame
c_galactic = c_icrs.galactic
c_galactic_hipparcos = c_icrs_hipparcos.galactic

# Add equivalent galactic coordinates back into data
data['pm_l_cosb'] = c_galactic.pm_l_cosb
data['pm_b'] = c_galactic.pm_b
confirmed_data['pm_l_cosb'] = c_galactic_hipparcos.pm_l_cosb
confirmed_data['pm_b'] = c_galactic_hipparcos.pm_b

# Sets distance limits
false_dist_indices = [i for i, x in enumerate(data['rest']) if x < 25 or x > 55]
data.remove_rows(false_dist_indices)

# Select stars within this data where pms are only in the region between pm_l = [-50,10] and pm_b = [-30,30]
#sel = data['pm_l_cosb'] >= -100
#sel &= data['pm_l_cosb'] < 250
#sel &= data['pm_b'] >= -50
#sel &= data['pm_b'] <= 200

sel = data['pmra'] >= -300
sel &= data['pmra'] < 300
sel &= data['pmdec'] >= -200
sel &= data['pmdec'] <= 200

small_area_stars = data[sel]

################## PLOTS PM PLOT AND DEFINES AREA OF INTEREST #################

## Plotting proper motion density plot
#fig = plt.figure()
#k = kde.gaussian_kde([small_area_stars['pm_l_cosb'], small_area_stars['pm_b']])
#nbins = 100
#xi, yi = np.mgrid[small_area_stars['pm_l_cosb'].min():small_area_stars['pm_l_cosb'].max():nbins*1j, small_area_stars['pm_b'].min():small_area_stars['pm_b'].max():nbins*1j]
#zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#cs = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_ncar_r)
#plt.colorbar()
#plt.xlabel('pm_l_cosb (mas/yr)')
#plt.ylabel('pm_b (mas/yr)')
#plt.title('Proper motion plot for area around BPMG members')
#plt.scatter(confirmed_data['pm_l_cosb'],confirmed_data['pm_b'], 0.1, 'k')
 
# Plotting proper motion density plot
fig = plt.figure()
k = kde.gaussian_kde([small_area_stars['pmra'], small_area_stars['pmdec']])
nbins = 100
xi, yi = np.mgrid[small_area_stars['pmra'].min():small_area_stars['pmra'].max():nbins*1j, small_area_stars['pmdec'].min():small_area_stars['pmdec'].max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
cs = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_ncar_r, vmax = 0.0001)
plt.colorbar()
plt.xlabel('pmra (mas/yr)')
plt.ylabel('pmdec (mas/yr)')
plt.title('Proper motion plot for area around BPMG members')
plt.scatter(confirmed_data['pmra_x'],confirmed_data['pmdec_x'], 0.1, 'k')

## Defines polygon vertices and path for area of interest
#verts = [
#        (10.,  40.),
#        (100.,-25.),
#        (160., 40.),
#        (160., 100.),
#        (110., 145.),
#        (25.,  110.),
#        (10.,  40.)
#        ]
#path = Path(verts)
#
## Overplots polygon enclosing area of interest
#ax = fig.add_subplot(111)
#patch = patches.PathPatch(path, lw=1, fill = False, color = 'r')
#ax.add_patch(patch)
#
## Determines which data points are inside area of interest
#points = np.column_stack((data['pm_l_cosb'],data['pm_b']))
#inside = path.contains_points(points)
#
#false_indices = [i for i, x in enumerate(inside) if not x]
#data.remove_rows(false_indices)


############################ PLOTS STAR POSITIONS ##############################
#
## Plots star positions
#plot_with_colourbar(data['ra'],data['dec'],data['phot_g_mean_mag'],'ra (deg)','dec (deg)','Location plot - OB2')
#
############################### PLOTS CAMDs ###################################
## Removes empty data from bp-rp
#masked_indices = [i for i, x in enumerate(data['bp_rp']) if np.ma.is_masked(x)]
#data.remove_rows(masked_indices)
#
## Converts Gaia g-band Magnitudes to Absolute G Band Magnitudes
#M_G = data['phot_g_mean_mag'] - 5*(np.log10(data['rest'])-1)
#
#bp_rp = data['bp_rp'] 
#mag_4_CAMD = M_G
#
## Plots Colour-Absolute Magnitude Diagram
##plot_with_colourbar(bp_rp,mag_4_CAMD,mag_4_CAMD,'BP-RP','Gaia Absolute G-band Magnitude','Colour-Absolute Magnitude Diagram for stars in the vicinity of the Hyades',invert_y_axis = True, y_lim = (15,-5))
#
## Plotting CAMD density plot
#fig2 = plt.figure()
#k2 = kde.gaussian_kde([bp_rp, mag_4_CAMD])
#nbins = 100
#x2i, y2i = np.mgrid[bp_rp.min():bp_rp.max():nbins*1j, mag_4_CAMD.min():mag_4_CAMD.max():nbins*1j]
#z2i = k2(np.vstack([x2i.flatten(), y2i.flatten()]))
#plt.gca().invert_yaxis()
#plt.gca().set_ylim(15,-5)
#cmap = plt.cm.gist_ncar_r
##cmaplist = [cmap(i) for i in range(cmap.N)]
##cmaplist[0] = (1.,1.,1.,1.0)
##cmap = cmap.from_list('Custom_cmap', cmaplist, cmap.N)
#cs2 = plt.pcolormesh(x2i, y2i, z2i.reshape(x2i.shape), cmap = cmap)
#plt.colorbar()
#plt.xlabel('BP-RP')
#plt.ylabel('Gaia Absolute G-band Magnitude')
#plt.title('Colour-Absolute Magnitude Density Plot for stars in the vicinity of the Hyades')

## Defines polygon vertices and path for area of interest
#verts_CAMD = [
#             (1.,  3.5),
#             (1.3, 6.),
#             (2.2, 8.),
#             (4.4, 14.8),
#             (4.9, 10.5),
#             (2.2, 4.5),
#             (1.,  3.5)
#             ]
#path_CAMD = Path(verts_CAMD)
#
## Overplots polygon enclosing area of interest
#ax = fig2.add_subplot(111)
#patch_CAMD = patches.PathPatch(path_CAMD, lw=1, fill = False, color = 'r')
#ax.add_patch(patch_CAMD)
#
####################### Re-plots PM Diagram for PMS stars ######################
#
## Determines which data points are inside area of interest
#points2 = np.column_stack((bp_rp,mag_4_CAMD))
#inside2 = path_CAMD.contains_points(points2)
#
#false_indices2 = [i for i, x in enumerate(inside2) if not x]
#data.remove_rows(false_indices2)
#
# Plotting proper motion density plot
#fig3 = plt.figure()
#k3 = kde.gaussian_kde([data['pm_l_cosb'], data['pm_b']])
#nbins = 100
#x3i, y3i = np.mgrid[data['pm_l_cosb'].min():data['pm_l_cosb'].max():nbins*1j, data['pm_b'].min():data['pm_b'].max():nbins*1j]
#z3i = k3(np.vstack([x3i.flatten(), y3i.flatten()]))
#cs3 = plt.pcolormesh(x3i, y3i, z3i.reshape(x3i.shape), cmap=plt.cm.gist_ncar_r)
#plt.colorbar()
#plt.xlabel('pm_l_cosb (mas/yr)')
#plt.ylabel('pm_b (mas/yr)')
#plt.title('Proper motion plot for potential Hyades members')
#plt.scatter(confirmed_data['pm_l_cosb'],confirmed_data['pm_b'], 0.1, 'k')
#
## Standard proper motion plot with colorbar representing distance
#plot_with_colourbar(data['pm_l_cosb'],data['pm_b'],data['rest'],'pm_l_cosb (mas/yr)','pm_b (mas/yr)','Proper motion plot for potential Hyades members','Distance (pc)')
#
## Final location plot
#plot_with_colourbar(data['ra'],data['dec'],data['phot_g_mean_mag'],'ra (deg)','dec (deg)','Location plot for potential Hyades members')
#
############################### Save final table ################################
##
#data.write('Reduced_Hyades_Data', format='votable')

stop = timeit.default_timer()
print('Time: ',stop - start)