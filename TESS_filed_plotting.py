#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:07:25 2020

TESS_field_plotting.py

Differen tools to plot TESS fields of view

@author: mbattley
"""

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as u

include_TOIs = True

cm = plt.cm.get_cmap('viridis')

#sector_list = ['S1','S2','S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13']

# Simple Scatter plot
data_S1_C4 = Table.read("exoplanet_archive_S1_C1.csv" , format='ascii.csv')
data = Table.read("complete_exoplanet_archive_simplified.csv", format='ascii.csv')
data_TOIs = Table.read("ALL_TOIs_22_6_20_xmatch_TIC.csv", format='ascii.csv')
data_S1_C4_TOIs = Table.read("S1C4_TOIs_22_6_20.csv" , format='ascii.csv')

mag_G = list(data_S1_C4['gaia_gmag'])
ra = list(data_S1_C4['ra'])
dec = list(data_S1_C4['dec'])
print('Length before: {}'.format(len(ra)))
if include_TOIs == True:
    ra = ra + list(data_S1_C4_TOIs['RA'])
    dec = dec + list(data_S1_C4_TOIs['Dec'])
    mag_G = mag_G + list(data_S1_C4_TOIs['Gmag'])
    print('Length after: {}'.format(len(ra)))

ra_C4_total = []
dec_C4_total = []
mag_G_C4_total = []

for i in range(len(data['ra'])):
    if data['S1'][i] == 4 or data['S2'][i] == 4 or data['S3'][i] == 4 or data['S4'][i] == 4 or data['S5'][i] == 4 or data['S6'][i] == 4 or data['S7'][i] == 4 or data['S8'][i] == 4 or data['S9'][i] == 4 or data['S10'][i]==4 or data['S11'][i]==4 or data['S12'][i]==4 or data['S13'][i]==4:
        ra_C4_total.append(data['ra'][i])
        dec_C4_total.append(data['dec'][i])
        mag_G_C4_total.append(data['gaia_gmag'][i])
print('Length full before: {}'.format(len(ra_C4_total)))
if include_TOIs == True:
    for i in range(len(data_TOIs['RA'])):
        if data_TOIs['S1'][i] == 4 or data_TOIs['S2'][i] == 4 or data_TOIs['S3'][i] == 4 or data_TOIs['S4'][i] == 4 or data_TOIs['S5'][i] == 4 or data_TOIs['S6'][i] == 4 or data_TOIs['S7'][i] == 4 or data_TOIs['S8'][i] == 4 or data_TOIs['S9'][i] == 4 or data_TOIs['S10'][i]==4 or data_TOIs['S11'][i]==4 or data_TOIs['S12'][i]==4 or data_TOIs['S13'][i]==4:
            ra_C4_total.append(data_TOIs['RA'][i])
            dec_C4_total.append(data_TOIs['Dec'][i])
            mag_G_C4_total.append(data_TOIs['Gmag'][i])
print('Length full before: {}'.format(len(ra_C4_total)))

c_equ_S1 = SkyCoord(ra, dec, unit='deg', frame='fk5')
c_ecl_S1=c_equ_S1.transform_to('geocentricmeanecliptic')
lon_S1 = c_ecl_S1.lon
lat_S1 = c_ecl_S1.lat

c_equ_full = SkyCoord(ra_C4_total, dec_C4_total, unit='deg', frame='fk5')
c_ecl_full=c_equ_full.transform_to('geocentricmeanecliptic')
lon_full = c_ecl_full.lon
lat_full = c_ecl_full.lat

################################# Equatorial ##################################
# All Southern Hemisphere sectors - Camera 4
eq_sc_plot = plt.figure()
plt.scatter(ra_C4_total[0:50],dec_C4_total[0:50],c=mag_G_C4_total[0:50],cmap=cm,s=mag_G_C4_total[0:50]*2,marker='o', label = 'planet')
plt.scatter(ra_C4_total[50:],dec_C4_total[50:],c=mag_G_C4_total[50:],cmap=cm,s=mag_G_C4_total[50:]*2,marker='^', label = 'TOI')
cbar = plt.colorbar()
cbar.set_label('Gaia G-Magnitude')
plt.clim(min(mag_G_C4_total), max(mag_G_C4_total))
plt.xlabel('ra (deg)')
plt.ylabel('dec (deg)')
plt.title('All TESS camera 4 planets and TOIs')
plt.legend()
plt.show()

# Sector 1 only
eq_S1_plot = plt.figure()
plt.scatter(ra[0:31],dec[0:31],c=mag_G[0:31],cmap=cm,s=mag_G[0:31]*2,marker = 'o', label='planet')
plt.scatter(ra[31:],dec[31:],c=mag_G[31:],cmap=cm,s=mag_G[31:]*2, marker='^', label = 'TOI')
cbar = plt.colorbar()
cbar.set_label('Gaia G-Magnitude')
plt.clim(min(mag_G), max(mag_G))
plt.xlabel('ra (deg)')
plt.ylabel('dec (deg)')
plt.title('TESS Camera 4 planets and TOIs, Sector 1 only')
plt.legend()
plt.show()


################################## Ecliptic ###################################
# All Southern Hemisphere sectors - Camera 4
ecl_sc_plot = plt.figure()
plt.scatter(lon_full[0:50],lat_full[0:50],c=mag_G_C4_total[0:50],cmap=cm,s=mag_G_C4_total[0:50]*2, marker='o', label='planet')
plt.scatter(lon_full[50:],lat_full[50:],c=mag_G_C4_total[50:],cmap=cm,s=mag_G_C4_total[50:]*2, marker='^', label='TOI')
cbar = plt.colorbar()
cbar.set_label('Gaia G-Magnitude')
plt.clim(min(mag_G_C4_total), max(mag_G_C4_total))
plt.xlabel('Ecliptic lon (deg)')
plt.ylabel('Ecliptic lat (deg)')
plt.title('All TESS camera 4 planets and TOIs')
plt.legend()
plt.show()


# Sector 1 only
ecl_sc_S1_plot = plt.figure()
plt.scatter(lon_S1[0:31],lat_S1[0:31],c=mag_G[0:31],cmap=cm,s=mag_G[0:31]*2, marker='o', label='planet')
plt.scatter(lon_S1[31:],lat_S1[31:],c=mag_G[31:],cmap=cm,s=mag_G[31:]*2, marker='^', label='TOI')
cbar = plt.colorbar()
cbar.set_label('Gaia G-Magnitude')
plt.clim(min(mag_G), max(mag_G))
plt.xlabel('Ecliptic lon (deg)')
plt.ylabel('Ecliptic lat (deg)')
plt.title('TESS Camera 4 planets and TOIs, Sector 1 only')
plt.legend()
plt.show()

## Hammer-Aitoff
#aitoff_fig = plt.figure()
#ax = aitoff_fig.add_subplot(111, projection='aitoff')
#ra = coord.Angle(data['ra'].filled(np.nan)*u.degree)
#ra = ra.wrap_at(180*u.degree)
#dec = coord.Angle(data['dec'].filled(np.nan)*u.degree)
#
#ax.scatter(ra.radian, dec.radian, s = 5,c=mag_G)
#plt.colorbar()