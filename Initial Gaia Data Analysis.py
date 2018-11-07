# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:45:22 2018
Opening Gaia data

This script loads in Gaia data from downloaded VOTables, before extracting the data arrays for plotting. 
@author: Matthew Battley
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import timeit

start = timeit.default_timer()
"""
# Reads votable from file
from astropy.io.votable import parse
votable_1 = parse("OB2_Box_1.vot")
table_1 = votable_1.get_first_table()
"""

def plot_with_colourbar(x,y,xlabel,ylabel,title,invert_y_axis = False):
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

#OR:
# Reads VOTable from file
from astropy.io.votable import parse_single_table
#table_1 = parse_single_table("OB2_Box_1.vot")
#table_2 = parse_single_table("OB2_Box_2.vot")
#table_3 = parse_single_table("OB2_Box_3.vot")
table_1 = parse_single_table("OB2_Box_1_bp-rp.vot")
table_2 = parse_single_table("OB2_Box_2_bp-rp.vot")
table_3 = parse_single_table("OB2_Box_3_bp-rp.vot")
#table_1 = parse_single_table("OB2_Box_1-TESS_Mag.vot")
#table_2 = parse_single_table("OB2_Box_2-TESS_Mag.vot")
#table_3 = parse_single_table("OB2_Box_3-TESS_Mag.vot")

data_1 = np.ma.filled(table_1.array['pmra'], np.NaN)

# Pulls values from data tables and combines all boxes into single parameter arrays
parameter_list = ['source_id', 'pmra', 'pmdec', 'ra', 'dec', 'phot_g_mean_mag', 'bp_rp', 'radial_velocity'] # Need to add RV and BP-RP colour as well
my_dict = {}

for i in parameter_list:
    my_dict[i] = np.concatenate((np.ma.filled(table_1.array[i], -2), np.ma.filled(table_2.array[i], -2),np.ma.filled(table_3.array[i], -2)))

source_id = my_dict['source_id'] # Source identification number
pmra = my_dict['pmra']           # Proper motion (right ascension)
pmdec = my_dict['pmdec']         # Proper motion (declination)
ra = my_dict['ra']               # ICRF Right Ascension
dec = my_dict['dec']             # ICRF Declination
mag = my_dict['phot_g_mean_mag'] # Photometric g-band magnitude
rv = my_dict['radial_velocity']  # Radial Velocity of Source
bp_rp = my_dict['bp_rp']         # BP-RP Colour index 

# Finds indices for the unique entries
unique_ids, indices = np.unique(source_id, return_index = True)
indices.sort()

# Obtains decreased arrays representing values for each unique ('u') star
u_pmra = pmra[indices]
u_pmdec = pmdec[indices]
u_ra = ra[indices]
u_dec = dec[indices]
u_mag = mag[indices]
u_rv = rv[indices]
u_bp_rp = bp_rp[indices]

# Sets up colourmap type, normalization and colours
cmap = matplotlib.cm.get_cmap('rainbow')
normalize = matplotlib.colors.Normalize(vmin = min(u_mag), vmax=max(u_mag))
colours = [cmap(normalize(value)) for value in u_mag]

# Plots proper motion (ra) vs proper motion (dec) graph
#fig_pm, ax = plt.subplots(figsize=(10,10))
#ax.scatter(u_pmra,u_pmdec,1,c=colours)
#plt.xlabel('pmra  (mas/yr)')
#plt.ylabel('pmdec  (mas/yr)')
#plt.title('Proper motion plot - OB2 - All Boxes')
#cax, _ = matplotlib.colorbar.make_axes(ax)
#cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm = normalize)
#cbar.set_label('g Magnitude')

# Plots star positions
fig_pos, ax = plt.subplots(figsize=(10,10))
#plt.scatter(u_ra,u_dec,0.5,c=colours)
#plt.xlabel('ra  (mas)')
#plt.ylabel('dec  (mas)')
#plt.title('Location plot - OB2 - All boxes')
#cax, _ = matplotlib.colorbar.make_axes(ax)
#cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm = normalize)
#cbar.set_label('g Magnitude')

# Plots Colour-Absolute Magnitude Diagram
#fig_CAMD, ax = plt.subplots(figsize=(10,10))
##real_indices = [i for i, x in enumerate(bp_rp) if x != -9999]
##u_bp_rp = u_bp_rp[real_indices] # Removes empty data from bp-rp
##mag_4_CAMD = u_mag[real_indices]
#plt.scatter(u_bp_rp,u_mag,0.5,c=colours)
#plt.gca().invert_yaxis()
#plt.xlabel('BP-RP')
#plt.ylabel('Absolute G Magnitude')
#plt.title('Colour-Absolute Magnitude Diagram for Stellar Association')
#cax, _ = matplotlib.colorbar.make_axes(ax)
#cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm = normalize)
#cbar.set_label('g Magnitude')

plot_with_colourbar(u_pmra,u_pmdec,'pmra (mas/yr)','pmdec (mas/yr)','Proper motion plot - OB2 - All Boxes',invert_y_axis = False)
plot_with_colourbar(u_ra,u_dec,'ra (mas)','dec (mas)','Location plot - OB2 - All boxes',invert_y_axis = False)
plot_with_colourbar(u_bp_rp,u_mag,'BP-RP','Absolute G Magnitude','Colour-Absolute Magnitude Diagram for Stellar Association OB2 - All Boxes',invert_y_axis = True)

# More things to add:
#   Lines on proper motion plot
#   Pull BP-RP and RV data from Gaia database
#   Collect BP and RP mean mag data
#   Colour-Absolute Magnitude Diagram (G_mag vs BP-RP)
#   UVW plots, two dimensions at a time 

stop = timeit.default_timer()

print('Time: ',stop - start)