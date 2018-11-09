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

def plot_with_colourbar(x,y,xlabel,ylabel,title,colours,cmap,invert_y_axis = False):
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

#filename_list = ["OB2_Box_1_bp-rp.vot","OB2_Box_2_bp-rp.vot","OB2_Box_3_bp-rp.vot"]
filename_list = ["OB2_Box_1-TESS_Mag.vot","OB2_Box_2-TESS_Mag.vot","OB2_Box_3-TESS_Mag.vot"]
tables = [0]*len(filename_list)
for i, x in enumerate(filename_list):
    tables[i] = parse_single_table(x)

data_1 = np.ma.filled(tables[0].array['pmra'], np.NaN)

# Pulls values from data tables and combines all boxes into single parameter arrays
parameter_list = ['source_id', 'pmra', 'pmdec', 'ra', 'dec', 'phot_g_mean_mag', 'bp_rp', 'radial_velocity'] # Need to add RV and BP-RP colour as well
my_dict = {}

for i in parameter_list:
    my_dict[i] = np.concatenate((np.ma.filled(tables[0].array[i], -999), np.ma.filled(tables[1].array[i], -999),np.ma.filled(tables[2].array[i], -999)))

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
plot_with_colourbar(u_pmra,u_pmdec,'pmra (mas/yr)','pmdec (mas/yr)','Proper motion plot - OB2 - All Boxes in TESS Mag range',colours,cmap,invert_y_axis = False)

# OR: Plotting with smoothed density plot:
#from scipy.stats import kde
#k = kde.gaussian_kde([u_pmra,u_pmdec])
#nbins = 300
#xi, yi = np.mgrid[u_pmra.min():u_pmra.max():nbins*1j, u_pmdec.min():u_pmdec.max():nbins*1j]
#zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)

# Plots star positions
plot_with_colourbar(u_ra,u_dec,'ra (mas)','dec (mas)','Location plot - OB2 - All boxes in TESS Mag range',colours,cmap,invert_y_axis = False)

real_indices = [i for i, x in enumerate(u_bp_rp) if x != -999]
u_bp_rp = u_bp_rp[real_indices] # Removes empty data from bp-rp
mag_4_CAMD = u_mag[real_indices]

cmap = matplotlib.cm.get_cmap('rainbow')
normalize = matplotlib.colors.Normalize(vmin = min(mag_4_CAMD), vmax=max(mag_4_CAMD))
colours = [cmap(normalize(value)) for value in mag_4_CAMD]

# Plots Colour-Absolute Magnitude Diagram
plot_with_colourbar(u_bp_rp,mag_4_CAMD,'BP-RP','Gaia G-band Magnitude','Colour-Absolute Magnitude Diagram for Stellar Association OB2 - All Boxes in TESS Mag range',colours,cmap,invert_y_axis = True)


# More things to add:
#   Lines on proper motion plot
#   UVW plots, two dimensions at a time 

stop = timeit.default_timer()

print('Time: ',stop - start)