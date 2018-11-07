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
import time

start_time = time.time()
"""
# Reads votable from file
from astropy.io.votable import parse
votable_1 = parse("OB2_Box_1.vot")
table_1 = votable_1.get_first_table()
"""

#OR:
# Reads VOTable from file
from astropy.io.votable import parse_single_table
table_1 = parse_single_table("OB2_Box_1.vot")
table_2 = parse_single_table("OB2_Box_2.vot")
table_3 = parse_single_table("OB2_Box_3.vot")

data_1 = np.ma.filled(table_1.array['pmra'], -99999)

# Pulls values from data tables and combines all boxes into single parameter arrays
parameter_list = ['source_id', 'pmra', 'pmdec', 'ra', 'dec', 'phot_g_mean_mag']
my_dict = {}

for i in parameter_list:
    my_dict[i] = np.concatenate((np.ma.filled(table_1.array[i], -99999), np.ma.filled(table_2.array[i], -99999),np.ma.filled(table_2.array[i], -99999)))

source_id = my_dict['source_id'] # Source identification number
pmra = my_dict['pmra']           # Proper motion (right ascension)
pmdec = my_dict['pmdec']         # Proper motion (declination)
ra = my_dict['ra']               # ICRF Right Ascension
dec = my_dict['dec']             # ICRF Declination
mag = my_dict['phot_g_mean_mag'] # Photometric g-band magnitude

# Finds indices for the unique entries
unique_ids, indices = np.unique(source_id, return_index = True)
indices.sort()

# Obtains decreased arrays representing values for each unique ('u') star
u_pmra = pmra[indices]
u_pmdec = pmdec[indices]
u_ra = ra[indices]
u_dec = dec[indices]
u_mag = mag[indices]

# Sets up colourmap type, normalization and colours
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin = min(u_mag), vmax=max(u_mag))
colors = [cmap(normalize(value)) for value in u_mag]

# Plots proper motion (ra) vs proper motion (dec) graph
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(u_pmra,u_pmdec,1,c=u_mag)
plt.xlabel('pmra  (mas/yr)')
plt.ylabel('pmdec  (mas/yr)')
plt.title('Proper motion plot - OB2 - All Boxes')
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm = normalize)
cbar.set_label('g Magnitude')

# Plots star positions
fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(u_ra,u_dec,0.5,c=u_mag)
plt.xlabel('ra  (mas)')
plt.ylabel('dec  (mas)')
plt.title('Location plot - OB2 - All boxes')
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm = normalize)
cbar.set_label('g Magnitude')

print('Total time: ', time.time() - start_time)