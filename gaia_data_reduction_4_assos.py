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
import pylab
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import timeit
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.stats import kde
from math import sin, cos

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

def spatial_plot_with_arrows(x, y, v1, v2, arrow_scale, title, xlabel, ylabel, xlim = False, ylim = False):
    """
    Function to plot quiver plots, when velocity info is desired alongside position
    """
    plt.figure()
    pylab.quiver(x, y, v1, v2, angles = 'uv', scale_units='xy', scale = arrow_scale)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim != False:
        plt.xlim(xlim)
    if ylim != False:
        plt.ylim(ylim)

def uvw(ra, dec, d, pmra, pmdec, rv):
    """
    Calculates galactic UVW velocities given ra, dec, distance, pmra, pmdec and RV
    
        PARAMETER - UNIT - DESCRIPTION
        ______________________________
    Inputs:
        ra        - deg    - Right Ascension
        dec       - deg    - Declination
        d         - pc     - Distance
        pmra      - mas/yr - Proper motion, Right Ascension
        pmdec     - mas/yr - Proper motion, Declination
        rv        - km/s   - Radial Velocity
    Outputs:
        U         - km/s   - Cartesian Galactic Velocity in X direction
                                Positive toward Galactic center
        V         - km/s   - Cartesian Galactic Velocity in Y direction
                                Positive in direction of Galactic rotation
        W         - km/s   - Cartesian Galactic Velocity in Z direction
                                Positive toward North Galactic Pole
    """
    
    # Initialise conversion constants
    k = 4.74047 # km/s equivalent of 1AU/yr
    A = [[ 0.0548755604,   0.8734370902,  0.4838350155],
         [ 0.4941094279, - 0.4448296300,  0.7469822445],
         [-0.8676661490, - 0.1980763734,  0.4559837762]]
    
    # Sets all parameters as arrays in case they were entered as lists
    ra = np.array(ra)
    dec = np.array(dec)
    d = np.array(d)
    pmra = np.array(pmra)
    pmdec = np.array(pmdec)
    rv = np.array(rv)
    
    # Precalculates trigonometric values
    cos_ra  = np.array([cos(i*np.pi/180) for i in ra])
    sin_ra  = np.array([sin(i*np.pi/180) for i in ra])
    cos_dec = np.array([cos(i*np.pi/180) for i in dec])
    sin_dec = np.array([sin(i*np.pi/180) for i in dec])
    
    # Set up velocities
    plx = 1000.0*1/d #parallax in mas
    vec1 = rv
    vec2 = k*pmra/plx
    vec3 = k*pmdec/plx
    
    # Calculate cartesian UVW velocities
    u = ( A[0][0]*cos_ra*cos_dec + A[0][1]*sin_ra*cos_dec + A[0][2]*sin_dec)*vec1 + \
        (-A[0][0]*sin_ra         + A[0][1]*cos_ra                          )*vec2 + \
        (-A[0][0]*cos_ra*sin_dec - A[0][1]*sin_ra*sin_dec + A[0][2]*cos_dec)*vec3
    v = ( A[1][0]*cos_ra*cos_dec + A[1][1]*sin_ra*cos_dec + A[1][2]*sin_dec)*vec1 + \
        (-A[1][0]*sin_ra         + A[1][1]*cos_ra                          )*vec2 + \
        (-A[1][0]*cos_ra*sin_dec - A[1][1]*sin_ra*sin_dec + A[1][2]*cos_dec)*vec3
    w = ( A[2][0]*cos_ra*cos_dec + A[2][1]*sin_ra*cos_dec + A[2][2]*sin_dec)*vec1 + \
        (-A[2][0]*sin_ra         + A[2][1]*cos_ra                          )*vec2 + \
        (-A[2][0]*cos_ra*sin_dec - A[2][1]*sin_ra*sin_dec + A[2][2]*cos_dec)*vec3
    u = -u # Reversing U so that it is +ve towards Galactic center
    
    return u,v,w

######################## IMPORTS AND SORTS OUT DATA ###########################

# Read data from table
Table = tab.Table
#data = Table.read('OB2_area_data.vot')
#confirmed_data = Table.read('OB2_Gaia_Zeeuw_Match_dist')
data = Table.read('Octans_area_data_dist.vot')
confirmed_data = Table.read('Bona-fide Octans members with dist.vot')

# Change from unrecognisable unit names in file
data['pmra'].unit = 'mas/yr'
data['pmdec'].unit = 'mas/yr'
data['radial_velocity'].unit = 'km/s'
confirmed_data['pmra'].unit = 'mas/yr'
confirmed_data['pmdec'].unit = 'mas/yr'
confirmed_data['ra'].unit = 'deg'
confirmed_data['dec'].unit = 'deg'

# Input sky coordinates for all stars
c_icrs = SkyCoord(ra = data['ra'], dec = data['dec'], pm_ra_cosdec = data['pmra'], pm_dec = data['pmdec'])
#c_icrs_confirmed = SkyCoord(ra = confirmed_data['ra'], dec = confirmed_data['dec'], pm_ra_cosdec = confirmed_data['pmra'], pm_dec = confirmed_data['pmdec'])
c_icrs_confirmed = SkyCoord(ra = confirmed_data['ra'], dec = confirmed_data['dec'], pm_ra_cosdec = confirmed_data['pmra'], pm_dec = confirmed_data['pmdec'])


# Convert star coordinates to Galactic frame
c_galactic = c_icrs.galactic
c_galactic_confirmed = c_icrs_confirmed.galactic

# Add equivalent galactic coordinates back into data
data['l'] = c_galactic.l
data['b'] = c_galactic.b
data['pm_l_cosb'] = c_galactic.pm_l_cosb
data['pm_b'] = c_galactic.pm_b

confirmed_data['l'] = c_galactic_confirmed.l
confirmed_data['b'] = c_galactic_confirmed.b
confirmed_data['pm_l_cosb'] = c_galactic_confirmed.pm_l_cosb
confirmed_data['pm_b'] = c_galactic_confirmed.pm_b

# Sets distance limits
false_dist_indices = [i for i, x in enumerate(data['rest']) if x >205 or x < 60 ]
data.remove_rows(false_dist_indices)

# Select stars within this data where pms are only near values for known members
#sel = data['pm_l_cosb'] >= -50
#sel &= data['pm_l_cosb'] < 10
#sel &= data['pm_b'] >= -30
#sel &= data['pm_b'] <= 30

sel = data['pmra'] >= -40
sel &= data['pmra'] < 40
sel &= data['pmdec'] >= -30
sel &= data['pmdec'] <= 50

small_area_stars = data[sel]

# Remove stars too close to background star population
#sel2 = data['pmra']

################## PLOTS PM PLOT AND DEFINES AREA OF INTEREST #################

## Plotting proper motion density plot (Galactic coords)
fig = plt.figure()
k = kde.gaussian_kde([small_area_stars['pm_l_cosb'], small_area_stars['pm_b']])
nbins = 100
xi, yi = np.mgrid[small_area_stars['pm_l_cosb'].min():small_area_stars['pm_l_cosb'].max():nbins*1j, small_area_stars['pm_b'].min():small_area_stars['pm_b'].max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
cs = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('pm_l_cosb (mas/yr)')
plt.ylabel('pm_b (mas/yr)')
plt.title('Proper motion plot for area around Octans members')
plt.scatter(confirmed_data['pm_l_cosb'],confirmed_data['pm_b'], 0.1, 'k')
 
# Plotting proper motion density plot (Equatorial Coords)
fig2 = plt.figure()
k = kde.gaussian_kde([small_area_stars['pmra'], small_area_stars['pmdec']])
nbins = 100
xi, yi = np.mgrid[small_area_stars['pmra'].min():small_area_stars['pmra'].max():nbins*1j, small_area_stars['pmdec'].min():small_area_stars['pmdec'].max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#cs = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_ncar_r, vmax = 0.000005)
cs = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('pmra (mas/yr)')
plt.ylabel('pmdec (mas/yr)')
plt.title('Proper motion plot for area around Octans')
plt.scatter(confirmed_data['pmra'],confirmed_data['pmdec'], 0.1, 'k')
#plt.scatter(small_area_stars['pm_l_cosb'],small_area_stars['pm_b'], 0.1, 'r')

## Defines polygon vertices and path for area of interest
verts = [
        (-31.,  7.3),
        (-14.5,-1.98),
        (-4.4, -12.2),
        (-17.5, -34.),
        (-41., -17.7),
        (-31.,  7.3),
        ]
path = Path(verts)

# Overplots polygon enclosing area of interest
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, lw=1, fill = False, color = 'r')
ax.add_patch(patch)

# Determines which data points are inside area of interest
points = np.column_stack((data['pm_l_cosb'],data['pm_b']))
inside = path.contains_points(points)

false_indices = [i for i, x in enumerate(inside) if not x]
data.remove_rows(false_indices)

############################ PLOTS STAR POSITIONS ##############################

# Plots star positions
plot_with_colourbar(data['ra'],data['dec'],data['phot_g_mean_mag'],'ra (deg)','dec (deg)','Location plot - Octans')

#################### CALCULATES GALACTIC POSITIONS AND VELOCITIES #############

ra_key = 'ra'
dec_key = 'dec'
pmra_key = 'pmra'
pmdec_key = 'pmdec'
d_key = 'rest'
rv_key = 'radial_velocity'

# Correcting for Inertial spin of Gaia DR2 proper motion system
omega_x = -0.086 # +/- 0.025 mas/yr
omega_y = -0.114 # +/- 0.025 mas/yr
omega_z = -0.037 # +/- 0.025 mas/yr

for i, dp in enumerate(data[pmra_key]):
    if data['phot_g_mean_mag'][i] <= 13:
        data[pmra_key][i]  = dp + omega_x*sin(data[dec_key][i]*np.pi/180)*cos(data[ra_key][i]*np.pi/180) + omega_y*sin(data[dec_key][i]*np.pi/180)*sin(data[ra_key][i]*np.pi/180) - omega_z*cos(data[dec_key][i]*np.pi/180)
        data[pmdec_key][i] = data[pmdec_key][i] - omega_x*sin(data[ra_key][i]*np.pi/180) + omega_y*cos(data[ra_key][i]*np.pi/180) 

# Calculate Galatic Positions
x_g = np.array([data[d_key][i] * cos(data['b'][i]*np.pi/180) * cos(data['l'][i]*np.pi/180) for i,d1 in enumerate(data['b'])]) 
y_g = np.array([data[d_key][i] * cos(data['b'][i]*np.pi/180) * sin(data['l'][i]*np.pi/180) for i,d1 in enumerate(data['b'])]) 
z_g = np.array([data[d_key][i] * sin(data['b'][i]*np.pi/180) for i,d1 in enumerate(data['b'])])

x_g_confirmed = np.array([confirmed_data[d_key][i] * cos(confirmed_data['b'][i]*np.pi/180) * cos(confirmed_data['l'][i]*np.pi/180) for i,d1 in enumerate(confirmed_data['b'])]) 
y_g_confirmed = np.array([confirmed_data[d_key][i] * cos(confirmed_data['b'][i]*np.pi/180) * sin(confirmed_data['l'][i]*np.pi/180) for i,d1 in enumerate(confirmed_data['b'])]) 
z_g_confirmed = np.array([confirmed_data[d_key][i] * sin(confirmed_data['b'][i]*np.pi/180) for i,d1 in enumerate(confirmed_data['b'])])

# Calculate Galactic Velocities
u_g, v_g, w_g = uvw(ra = data[ra_key], dec = data[dec_key], d = data['rest'], pmra = data[pmra_key], pmdec = data[pmdec_key], rv = data[rv_key])        # Hyades
u_g_confirmed, v_g_confirmed, w_g_confirmed = uvw(ra = confirmed_data[ra_key], dec = confirmed_data[dec_key], d = confirmed_data['rest'], pmra = confirmed_data[pmra_key], pmdec = confirmed_data[pmdec_key], rv = confirmed_data['radvel'])        # Hyades

#Plot position at current time with arrows for velocities
#spatial_plot_with_arrows(y_g, x_g, v_g, u_g, 1, 'Galactic XY Position of Hyades members, with overplotted UV velocities \n Current time', 'Y (pc)', 'X (pc)')
#spatial_plot_with_arrows(x_g, z_g, u_g, w_g, 1, 'Galactic XZ Position of Hyades members, with overplotted UW velocities \n Current time', 'X (pc)', 'Z (pc)')
#spatial_plot_with_arrows(y_g, z_g, v_g, w_g, 1, 'Galactic YZ Position of Hyades members, with overplotted VW velocities \n Current time', 'Y (pc)', 'Z (pc)')

# Density plot for XY Galactic position
XY_density_fig = plt.figure()
k = kde.gaussian_kde([x_g,y_g])
nbins = 100
x4i, y4i = np.mgrid[x_g.min():x_g.max():nbins*1j, y_g.min():y_g.max():nbins*1j]
z4i = k(np.vstack([x4i.flatten(), y4i.flatten()]))
cs4 = plt.pcolormesh(x4i, y4i, z4i.reshape(x4i.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('x_g')
plt.ylabel('y_g')
plt.title('Galactic XY plot for potential Octans members')
plt.scatter(x_g_confirmed,y_g_confirmed, 0.1, 'k')

# Density plot for XZ Galactic position
XZ_density_fig = plt.figure()
k = kde.gaussian_kde([x_g,z_g])
nbins = 100
x6i, y6i = np.mgrid[x_g.min():x_g.max():nbins*1j, z_g.min():z_g.max():nbins*1j]
z6i = k(np.vstack([x6i.flatten(), y6i.flatten()]))
cs6 = plt.pcolormesh(x6i, y6i, z6i.reshape(x6i.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('x_g')
plt.ylabel('z_g')
plt.title('Galactic XZ plot for potential Octans members')
plt.scatter(x_g_confirmed,z_g_confirmed, 0.1, 'k')

# Density plot for YZ Galactic position
YZ_density_fig = plt.figure()
k = kde.gaussian_kde([y_g,z_g])
nbins = 100
x5i, y5i = np.mgrid[y_g.min():y_g.max():nbins*1j, z_g.min():z_g.max():nbins*1j]
z5i = k(np.vstack([x5i.flatten(), y5i.flatten()]))
cs5 = plt.pcolormesh(x5i, y5i, z5i.reshape(x5i.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('y_g')
plt.ylabel('z_g')
plt.title('Galactic YZ plot for potential Octans members')
plt.scatter(y_g_confirmed,z_g_confirmed, 0.1, 'k')

######### POLYGON METHOD OF NARROWING DOWN ####################################
## Defines polygon vertices and path for area of interest
#verts = [
#        (-137., -24.0),
#        (-88,-29.4),
#        (-18, -50.),
#        (-18, -78.),
#        (-78., -103.0),
#        (-192.,-60.),
#        (-193.,-40.),
#        (-137., -24.0),
#        ]
#path = Path(verts)
#
## Overplots polygon enclosing area of interest
#ax = YZ_density_fig.add_subplot(111)
#patch = patches.PathPatch(path, lw=1, fill = False, color = 'r')
#ax.add_patch(patch)
#
## Determines which data points are inside area of interest
#points = np.column_stack((y_g,z_g))
#inside = path.contains_points(points)
#
#false_indices = [i for i, x in enumerate(inside) if not x]
#data.remove_rows(false_indices)
################# MEAN + STANDARD DEV. METHOD OF NARROWING DOWN ###############

x_g_mean = np.mean(x_g_confirmed)
y_g_mean = np.mean(y_g_confirmed)
z_g_mean = np.mean(z_g_confirmed)

x_g_std = np.std(x_g_confirmed)
y_g_std = 28.33
z_g_std = np.std(z_g_confirmed)


good_indices_1 = np.where(np.logical_and(x_g>=x_g_mean - 3*x_g_std, x_g <= x_g_mean + 3*x_g_std))
good_indices_2 = np.where(np.logical_and(y_g>=y_g_mean - 3*y_g_std, y_g <= y_g_mean + 3*y_g_std))
good_indices_3 = np.where(np.logical_and(z_g>=z_g_mean - 3*z_g_std, z_g <= z_g_mean + 3*z_g_std))

full_good_indices = np.intersect1d(good_indices_2, good_indices_3)
data = data[full_good_indices]

###############################################################################

# Recalculate XYZ positions
x_g = np.array([data[d_key][i] * cos(data['b'][i]*np.pi/180) * cos(data['l'][i]*np.pi/180) for i,d1 in enumerate(data['b'])]) 
y_g = np.array([data[d_key][i] * cos(data['b'][i]*np.pi/180) * sin(data['l'][i]*np.pi/180) for i,d1 in enumerate(data['b'])]) 
z_g = np.array([data[d_key][i] * sin(data['b'][i]*np.pi/180) for i,d1 in enumerate(data['b'])])

# Recalculate UVW velocities
u_g, v_g, w_g = uvw(ra = data[ra_key], dec = data[dec_key], d = data['rest'], pmra = data[pmra_key], pmdec = data[pmdec_key], rv = data[rv_key])        # Hyades

## Density plot for YZ Galactic position
#YZ_density_fig = plt.figure()
#k = kde.gaussian_kde([y_g,z_g])
#nbins = 100
#x5i, y5i = np.mgrid[y_g.min():y_g.max():nbins*1j, z_g.min():z_g.max():nbins*1j]
#z5i = k(np.vstack([x5i.flatten(), y5i.flatten()]))
#cs5 = plt.pcolormesh(x5i, y5i, z5i.reshape(x5i.shape), cmap=plt.cm.viridis)
#plt.colorbar()
#plt.xlabel('y_g')
#plt.ylabel('z_g')
#plt.title('Galactic YZ plot after 2nd polygon selection')
#plt.scatter(y_g_confirmed,z_g_confirmed, 0.1, 'k')
#
## Density plot for XY Galactic position
#XY_density_fig = plt.figure()
#k = kde.gaussian_kde([x_g,y_g])
#nbins = 100
#x4i, y4i = np.mgrid[x_g.min():x_g.max():nbins*1j, y_g.min():y_g.max():nbins*1j]
#z4i = k(np.vstack([x4i.flatten(), y4i.flatten()]))
#cs4 = plt.pcolormesh(x4i, y4i, z4i.reshape(x4i.shape), cmap=plt.cm.viridis)
#plt.colorbar()
#plt.xlabel('x_g')
#plt.ylabel('y_g')
#plt.title('Galactic XY plot for potential Octans members')
#plt.scatter(x_g_confirmed,y_g_confirmed, 0.1, 'k')
#
## Density plot for XZ Galactic position
#XZ_density_fig = plt.figure()
#k = kde.gaussian_kde([x_g,z_g])
#nbins = 100
#x6i, y6i = np.mgrid[x_g.min():x_g.max():nbins*1j, z_g.min():z_g.max():nbins*1j]
#z6i = k(np.vstack([x6i.flatten(), y6i.flatten()]))
#cs6 = plt.pcolormesh(x6i, y6i, z6i.reshape(x6i.shape), cmap=plt.cm.viridis)
#plt.colorbar()
#plt.xlabel('x_g')
#plt.ylabel('z_g')
#plt.title('Galactic XZ plot for potential Octans members')
#plt.scatter(x_g_confirmed,z_g_confirmed, 0.1, 'k')

nancut = np.isnan(u_g)
u_g = u_g[~nancut]
v_g = v_g[~nancut]
w_g = w_g[~nancut]

nancut_confirmed = np.isnan(u_g_confirmed)
u_g_confirmed = u_g_confirmed[~nancut_confirmed]
v_g_confirmed = v_g_confirmed[~nancut_confirmed]
w_g_confirmed = w_g_confirmed[~nancut_confirmed]

# Density plot for VW Galactic Velocity
VW_density_fig = plt.figure()
k = kde.gaussian_kde([v_g,w_g])
nbins = 100
x5i, y5i = np.mgrid[v_g.min():v_g.max():nbins*1j, w_g.min():w_g.max():nbins*1j]
z5i = k(np.vstack([x5i.flatten(), y5i.flatten()]))
cs5 = plt.pcolormesh(x5i, y5i, z5i.reshape(x5i.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('V_g (km/s)')
plt.ylabel('W_g (km/s)')
plt.title('Galactic VW plot after 2nd cut')
plt.scatter(v_g_confirmed,w_g_confirmed, 0.1, 'k')

# Density plot for UV Galactic Velocity
UV_density_fig = plt.figure()
k = kde.gaussian_kde([u_g,v_g])
nbins = 100
x4i, y4i = np.mgrid[u_g.min():u_g.max():nbins*1j, v_g.min():v_g.max():nbins*1j]
z4i = k(np.vstack([x4i.flatten(), y4i.flatten()]))
cs4 = plt.pcolormesh(x4i, y4i, z4i.reshape(x4i.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('U_g')
plt.ylabel('V_g')
plt.title('Galactic UV plot after 2nd cut')
plt.scatter(u_g_confirmed,v_g_confirmed, 0.1, 'k')

# Density plot for UW Galactic position
UW_density_fig = plt.figure()
k = kde.gaussian_kde([u_g,w_g])
nbins = 100
x6i, y6i = np.mgrid[u_g.min():u_g.max():nbins*1j, w_g.min():w_g.max():nbins*1j]
z6i = k(np.vstack([x6i.flatten(), y6i.flatten()]))
cs6 = plt.pcolormesh(x6i, y6i, z6i.reshape(x6i.shape), cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('U_g')
plt.ylabel('W_g')
plt.title('Galactic UW plot after 2nd cut')
plt.scatter(u_g_confirmed,w_g_confirmed, 0.1, 'k')

plot_with_colourbar(data['ra'],data['dec'],data['phot_g_mean_mag'],'ra (deg)','dec (deg)','Location plot after XYZ cut - Octans')

##############

u_g_mean = np.mean(u_g_confirmed)
v_g_mean = np.mean(v_g_confirmed)
w_g_mean = np.mean(w_g_confirmed)

u_g_std = np.std(u_g_confirmed)
v_g_std = np.std(v_g_confirmed)
w_g_std = np.std(w_g_confirmed)


good_indices_1 = np.where(np.logical_and(u_g>=u_g_mean - 5*u_g_std, u_g <= u_g_mean + 5*u_g_std))
good_indices_2 = np.where(np.logical_and(v_g>=v_g_mean - 5*v_g_std, v_g <= v_g_mean + 5*v_g_std))
good_indices_3 = np.where(np.logical_and(w_g>=w_g_mean - 5*w_g_std, w_g <= w_g_mean + 5*w_g_std))

half_good_indices = np.intersect1d(good_indices_1, good_indices_2)
full_good_indices = np.intersect1d(half_good_indices, good_indices_3)
data = data[full_good_indices]

plot_with_colourbar(data['ra'],data['dec'],data['phot_g_mean_mag'],'ra (deg)','dec (deg)','Location plot after all cuts - Octans')

############################### PLOTS CAMDs ###################################
# Removes empty data from bp-rp
#masked_indices = [i for i, x in enumerate(data['bp_rp']) if np.ma.is_masked(x)]
#data.remove_rows(masked_indices)
#
# Converts Gaia g-band Magnitudes to Absolute G Band Magnitudes
M_G = data['phot_g_mean_mag'] - 5*(np.log10(data['rest'])-1)

bp_rp = data['bp_rp'] 
mag_4_CAMD = M_G

nancut = np.isnan(bp_rp)
bp_rp = bp_rp[~nancut]
mag_4_CAMD = mag_4_CAMD[~nancut]

# Plots Colour-Absolute Magnitude Diagram
#plot_with_colourbar(bp_rp,mag_4_CAMD,mag_4_CAMD,'BP-RP','Gaia Absolute G-band Magnitude','Colour-Absolute Magnitude Diagram for stars in the vicinity of the Hyades',invert_y_axis = True, y_lim = (15,-5))

# Plotting CAMD density plot
fig3 = plt.figure()
k2 = kde.gaussian_kde([bp_rp, mag_4_CAMD])
nbins = 500
x2i, y2i = np.mgrid[bp_rp.min():bp_rp.max():nbins*1j, mag_4_CAMD.min():mag_4_CAMD.max():nbins*1j]
z2i = k2(np.vstack([x2i.flatten(), y2i.flatten()]))
plt.gca().invert_yaxis()
plt.gca().set_ylim(15,-5)
cmap = plt.cm.viridis
#cmaplist = [cmap(i) for i in range(cmap.N)]
#cmaplist[0] = (1.,1.,1.,1.0)
#cmap = cmap.from_list('Custom_cmap', cmaplist, cmap.N)
cs2 = plt.pcolormesh(x2i, y2i, z2i.reshape(x2i.shape), cmap = cmap)
plt.colorbar()
plt.xlabel('BP-RP')
plt.ylabel('Gaia Absolute G-band Magnitude')
plt.title('Colour-Absolute Magnitude Density Plot for stars in the vicinity of Octans')

print('Length of data is now: {}'.format(len(data['ra'])))

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
##
### Overplots polygon enclosing area of interest
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
#pm_cut_gal_fig = plt.figure()
#k3 = kde.gaussian_kde([data['pm_l_cosb'], data['pm_b']])
#nbins = 100
#x3i, y3i = np.mgrid[data['pm_l_cosb'].min():data['pm_l_cosb'].max():nbins*1j, data['pm_b'].min():data['pm_b'].max():nbins*1j]
#z3i = k3(np.vstack([x3i.flatten(), y3i.flatten()]))
#cs3 = plt.pcolormesh(x3i, y3i, z3i.reshape(x3i.shape), cmap=plt.cm.viridis)
#plt.colorbar()
#plt.xlabel('pm_l_cosb (mas/yr)')
#plt.ylabel('pm_b (mas/yr)')
#plt.title('Proper motion plot for potential Octans members')
#plt.scatter(confirmed_data['pm_l_cosb'],confirmed_data['pm_b'], 0.1, 'k')
#
## Plotting proper motion density plot (Equatorial Coords)
#pm_cut_equat_fig = plt.figure()
#k = kde.gaussian_kde([data['pmra'], data['pmdec']])
#nbins = 100
#xi, yi = np.mgrid[data['pmra'].min():data['pmra'].max():nbins*1j, data['pmdec'].min():data['pmdec'].max():nbins*1j]
#zi = k(np.vstack([xi.flatten(), yi.flatten()]))
##cs = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_ncar_r, vmax = 0.000005)
#cs = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.viridis)
#plt.colorbar()
#plt.xlabel('pmra (mas/yr)')
#plt.ylabel('pmdec (mas/yr)')
#plt.title('Proper motion plot for potential Octans members')
#plt.scatter(confirmed_data['pmra'],confirmed_data['pmdec'], 0.1, 'k')

## Standard proper motion plot with colorbar representing distance
#plot_with_colourbar(data['pm_l_cosb'],data['pm_b'],data['rest'],'pm_l_cosb (mas/yr)','pm_b (mas/yr)','Proper motion plot for potential Hyades members','Distance (pc)')
#
## Final location plot
#plot_with_colourbar(data['ra'],data['dec'],data['phot_g_mean_mag'],'ra (deg)','dec (deg)','Location plot for potential Hyades members')
#
############################### Save final table ################################
##
#with open('Reduced_Octans_Data', 'w+') as f:
#    f.write(data, format = 'votable')
data.write('/home/astro/phrhzn/Documents/PhD/Association Members/Octans'+'Reduced_Octans_Data', format='votable')

stop = timeit.default_timer()
print('Time: ',stop - start)