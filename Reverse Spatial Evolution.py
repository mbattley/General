#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:26:10 2018

This code investigates the reverse evolution of clusters by running their 
current proper motions back in time by the approximate age of the system.

@author: Matthew Battley
"""

import astropy.table as tab
import numpy as np
import matplotlib.pyplot as plt
import pylab
import timeit
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from math import sin, cos
from matplotlib import animation
from scipy.optimize import minimize

start = timeit.default_timer()

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

def old_position_angle(current_ra, current_dec, pmra, pmdec, age):
    """
    Calculates the old ra/dec positions of stars given their current positions,
    velcoities and approximate ages
    
        PARAMETER   - UNIT
        __________________
    Inputs:
        current_ra  - deg
        current_dec - deg
        pmra        - mas/yr = deg/Myr
        pmdec       - mas/yr = deg/Myr
        age         - Myr
    Outputs:
        old_ra      - deg
        old_dec     - deg
    
    """
    #nb units = mas/yr * Myr = deg/Myr * Myr = deg 
    old_ra = current_ra - pmra*age 
    old_dec = current_dec - pmdec*age
    
    return old_ra, old_dec

def print_spectral_types(spectral_type):
    """
    Sums up total and prints out spectral type distribution for all stars in a 
    given table
    """
    
    O, B, A, F, G, K, M = 0, 0, 0, 0, 0, 0, 0
    
    for i, x in enumerate(spectral_type):
        if spectral_type[i] == '':
            pass
        elif spectral_type[i][0] == 'O':
            O += 1
        elif spectral_type[i][0] == 'B':
            B += 1
        elif spectral_type[i][0] == 'A':
            A += 1
        elif spectral_type[i][0] == 'F':
            F += 1
        elif spectral_type[i][0] == 'G':
            G += 1
        elif spectral_type[i][0] == 'K':
            K += 1
        elif spectral_type[i][0] == 'M':
            M += 1

    print("O: {}, B: {}, A: {}, F: {}, G: {}, K: {}, M: {}".format(O,B,A,F,G,K,M))

def convergent_point(filename):
    """
    Carries out convergent point method to determine the position of the 
    convergent point for a given star cluster or association, along with its 
    respective probability of belonging to the group of interest.
    
    Based on de Bruijne's (1999) Convergent Point Method
    
        PARAMETER  - UNIT  - DESCRIPTION
        ________________________________
    Inputs:
        group_data -  NA   - Input data table for the group
    Outputs:
        cp_ra      -  deg  - RA position of convergent point  
        cp_dec     -  deg  - Dec position of convergent point
        ra         -  deg  - Array of RA positions for each star used
        dec        -  deg  - Array of Dec positions for each star used
        p_cp       -   %   - Array of probabilities for each star belonging
    """
    # Method Psuedo-code:
    # 1. Initialise stuff
    # 1.5 Discard stars with insignificant proper motions
    # 2. Set initial cp positions
    # 3. Calculate theta and sigma_theta for every star based on cp position (at once, in array)
    # 4. Calculate mu_ll and mu_t for each star (at once, in array)
    # 5. Calculate t_t and t_t**2 for each star (at once, in array)
    # 6. Sum t_t**2 column and record in new results array (along with ra_cp and dec_cp)
    # 7. Repeat for all cp positions
    # 8. Find smallest t_t**2 and hence true cp position
    # 9. Recalculate t_t for all stars
    # 10. Use this to find probability for each star belonging
    
    group_data = Table.read(filename)
    
#    t_min = 5
#    sigma_int = 10 #mas/yr - expected internal proper motion distribution
    
#    use = np.array([True]*len(group_data[ra_key]))
    
#    mu = np.sqrt(group_data[pmra_key]**2 + group_data[pmdec_key]**2) #Total proper motion
#    sigma_mu = np.sqrt(group_data['pmra_error']**2 + group_data['pmdec_error']**2) #Error in total proper motion
    
#    t = mu/np.sqrt(sigma_mu**2 + sigma_int**2) #Eq (5) from de Zeeuw et al., 1999
    
    # Determining stars with insignificant proper motions
#    for i, t_val in enumerate(t):
#        if t_val <= t_min:
#            use[i] = False
    
#    false_indices = [i for i, x in enumerate(use) if not x]
    
    # Removes stars with insiginificant proper motions from cp calculation
#    group_data.remove_rows(false_indices)
    
    ra_cp = list(range(0,361))
    dec_cp = list(range(-90,91))
    
    t_t_squared_array = np.zeros((361,181))
    
    sigma_theta = np.arctan(np.sin((np.array(group_data['ra_error']))*np.pi/180)/(-np.sin(np.array(group_data['dec_error'])*np.pi/180)*np.cos((np.array(group_data['ra_error']))*np.pi/180)))
    
    # Compute chi-squared array for every possible cp position
    for i in list(range(0,len(ra_cp))):
        for j in list(range(len(dec_cp))):
            theta = np.arctan(np.sin((ra_cp[i] - group_data[ra_key])*np.pi/180)/(np.cos(group_data[dec_key]*np.pi/180)*np.tan(dec_cp[j]*np.pi/180)-np.sin(group_data[dec_key]*np.pi/180)*np.cos((ra_cp[i]-group_data[ra_key])*np.pi/180)))
            
            mu_ll = np.sin(theta)*group_data[pmra_key] + np.cos(theta)*group_data[pmdec_key]
            mu_t = -np.cos(theta)*group_data[pmra_key] + np.sin(theta)*group_data[pmdec_key]
            sigma_t_squared = (sigma_theta*mu_ll)**2 + (group_data['pmra_error']*np.cos(theta))**2 + (group_data['pmdec_error']*np.sin(theta))**2
            
            t_t_squared = mu_t**2/sigma_t_squared
    
            t_t_squared_array[i][j] = t_t_squared.sum()
     
    # Determines true cp (which minimises chi-squared)       
    cp_pos = np.unravel_index(t_t_squared_array.argmin(), t_t_squared_array.shape)
    cp_ra = cp_pos[0]
    cp_dec = cp_pos[1]-90
    
    # Determines true t-tangent
    theta = np.arctan(np.sin((cp_ra - group_data[ra_key])*np.pi/180)/(np.cos(group_data[dec_key]*np.pi/180)*np.tan(cp_dec*np.pi/180)-np.sin(group_data[dec_key]*np.pi/180)*np.cos((cp_ra-group_data[ra_key])*np.pi/180)))
           
    mu_ll = np.sin(theta)*group_data[pmra_key] + np.cos(theta)*group_data[pmdec_key]
    mu_t = -np.cos(theta)*group_data[pmra_key] + np.sin(theta)*group_data[pmdec_key]
    sigma_t_squared = (sigma_theta*mu_ll)**2 + (group_data['pmra_error']*np.cos(theta))**2 + (group_data['pmdec_error']*np.sin(theta))**2
            
    true_t_t_squared = mu_t**2/sigma_t_squared
    
    # Calculates probability for each star belonging
    p_cp = np.exp(-0.5*true_t_t_squared) 
    p_cp.name = 'p_cp'
    
    return cp_ra, cp_dec, group_data[ra_key], group_data[dec_key], p_cp

def spatial_plot(x, y, spot_scale, title, xlabel, ylabel, xlim = False, ylim = False):
    """
    Function to plot standard scatterplots
    """
    plt.figure()
    plt.scatter(x, y, spot_scale)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim != False:
        plt.xlim(xlim)
    if ylim != False:
        plt.ylim(ylim)
        
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


####################### DATA GATHERING & PREPARATION ##########################

# Read data from table
Table = tab.Table
#table_data = Table.read('Hipparcos_OB2_de_Zeeuw_1999.vot')
#table_data = Table.read('OB2_Gaia_Zeeuw_Match_dist')
#table_data = Table.read('Pleiades_data_GaiaDR2')
#table_data = Table.read('Hyades_data')
table_data = Table.read('Reduced_Hyades_Data_with_RV')

# Allows user to specify different names for information rows 
ra_key = 'ra'
dec_key = 'dec'
pmra_key = 'pmra'
pmdec_key = 'pmdec'
d_key = 'rest'
rv_key = 'radvel'
#plx_key = 'parallax'

# Change from unrecognisable unit names in file
table_data[pmra_key].unit = 'mas/yr'
table_data[pmdec_key].unit = 'mas/yr'
table_data[ra_key].unit = 'deg'
table_data[dec_key].unit = 'deg'
table_data[rv_key].unit = 'km/s'

# Assembles age matrix (in this case, quite approximate, broken only into Sco_Cen groups)
#table_data['age'] = np.array(['None']*len(table_data['ra_1'])) # OB2
#table_data['age'] = np.array([100]*len(table_data[pmra_key])) #Pleiades age ~100 Mya
table_data['age'] = np.array([625]*len(table_data[pmra_key])) #Reduced Hyades
#US_age = 11 # Myr
#UCL_age = 16 # Myr
#LCC_age = 17 #Myr
#US_age = 17 # Myr
#UCL_age = 17 # Myr
#LCC_age = 17 #Myr
#
#for i, data in enumerate(table_data['OBAss']):
#    if data == 'A':
#        table_data['age'][i] = US_age
#    elif data == 'B':
#        table_data['age'][i] = UCL_age
#    else:
#        table_data['age'][i] = LCC_age

ages = [float(i) for i in table_data['age']] # (Myr)

# Input sky coordinates for all stars
#c_icrs_hipparcos = SkyCoord(ra = table_data['ra_1'], dec = table_data['dec_1'], pm_ra_cosdec = table_data['pmra_1'], pm_dec = table_data['pmdec_1'])
c_icrs_hipparcos = SkyCoord(ra = table_data[ra_key], dec = table_data[dec_key], pm_ra_cosdec = table_data[pmra_key], pm_dec = table_data[pmdec_key])

# Convert star coordinates to Galactic frame
c_galactic_hipparcos = c_icrs_hipparcos.galactic
#print(c_galactic_hipparcos)

# Add equivalent galactic coordinates back into data
"""
You can also get l and b straight from Gaia data... worth checking if they compare
"""
table_data['l'] = c_galactic_hipparcos.l
table_data['b'] = c_galactic_hipparcos.b
table_data['pm_l_cosb'] = c_galactic_hipparcos.pm_l_cosb
table_data['pm_b'] = c_galactic_hipparcos.pm_b

# Correcting for Inertial spin of Gaia DR2 proper motion system
omega_x = -0.086 # +/- 0.025 mas/yr
omega_y = -0.114 # +/- 0.025 mas/yr
omega_z = -0.037 # +/- 0.025 mas/yr

for i, dp in enumerate(table_data[pmra_key]):
    if table_data['phot_g_mean_mag'][i] <= 13:
        table_data[pmra_key][i]  = dp + omega_x*sin(table_data[dec_key][i]*np.pi/180)*cos(table_data[ra_key][i]*np.pi/180) + omega_y*sin(table_data[dec_key][i]*np.pi/180)*sin(table_data[ra_key][i]*np.pi/180) - omega_z*cos(table_data[dec_key][i]*np.pi/180)
        table_data[pmdec_key][i] = table_data[pmdec_key][i] - omega_x*sin(table_data[ra_key][i]*np.pi/180) + omega_y*cos(table_data[ra_key][i]*np.pi/180) 


######################## GALACTIC LAT/LONG PLOTS ##############################

# Calculates approximate initial distribution of OB2 members given approximate association age
original_l, original_b = old_position_angle(table_data['l'], table_data['b'], table_data['pm_l_cosb'],table_data['pm_b'],ages)

# Compensates for going around multiple times
for i, data in enumerate(original_l):
    if data >= 360:
        original_l[i] = data - 360*(data//360)
#    if data < 0:
#        original_l[i] = data + 360*(data//360)
#
#for i, data in enumerate(original_b):
#    if data > 90:
#        original_b[i] = data - 180*(1+data//180)
#    if data < -90:
#        original_b[i] = data + 180*(1+data//180)

# Plots positions at current time
#spatial_plot(table_data['l'], table_data['b'], 1, 'Position of confirmed Hyades members at current time', 'l (deg)', 'b (deg)')

# Plots positions of association stars near formation
#spatial_plot(original_l, original_b, 2, 'Position of confirmed OB2 Association members near splitting time \n Age ={0} Myr'.format(table_data['age'][0]), 'l (deg)', 'b (deg)')

# Plot figure with arrows for velocities
#spatial_plot_with_arrows(table_data['l'], table_data['b'], table_data['pm_l_cosb'], table_data['pm_b'], 40, 'Position and Velocities of OB2 members at current time', 'l (deg)', 'b (deg)')

######################### CONVERGENT POINT METHOD #############################
#####################(Based on de Bruijne 1999 method)#########################
# Method Psuedo-code:
# 1. Initialise stuff
# 1.5 Discard stars with insignificant proper motions
# 2. Set initial cp positions
# 3. Calculate theta and sigma_theta for every star based on cp position (at once, in array)
# 4. Calculate mu_ll and mu_t for each star (at once, in array)
# 5. Calculate t_t and t_t**2 for each star (at once, in array)
# 6. Sum t_t**2 column and record in new results array (along with ra_cp and dec_cp)
# 7. Repeat for all cp positions
# 8. Find smallest t_t**2 and hence true cp position
# 9. Recalculate t_t for all stars
# 10. Use this to find probability for each star belonging

#t_min = 5
#sigma_int = 10 #mas/yr - expected internal proper motion distribution
#
#use = np.array([True]*len(table_data[ra_key]))
#
#mu = np.sqrt(table_data[pmra_key]**2 + table_data[pmdec_key]**2) #Total proper motion
#sigma_mu = np.sqrt(table_data['pmra_error']**2 + table_data['pmdec_error']**2) #Error in total proper motion
#
#t = mu/np.sqrt(sigma_mu**2 + sigma_int**2) #Eq (5) from de Zeeuw et al., 1999
#
## Determining stars with insignificant proper motions
#for i, t_val in enumerate(t):
#    if t_val <= t_min:
#        use[i] = False
#
#use_indices = [i for i, x in enumerate(use) if x]
#
## Removes stars with insiginificant proper motions from cp calculation
##ra4cp = table_data[ra_key][use_indices]
##dec4cp = table_data[dec_key][use_indices]
##ra_err4cp = table_data['ra_error'][use_indices]
##dec_err4cp = table_data['dec_error'][use_indices]
##
##pmra4cp = 
#
#ra_cp = list(range(0,360))
#dec_cp = list(range(-90,90))
#
#t_t_squared_array = np.zeros((360,180))
#
#sigma_theta = np.arctan(np.sin((np.array(table_data['ra_error']))*np.pi/180)/(-np.sin(np.array(table_data['dec_error'])*np.pi/180)*np.cos((np.array(table_data['ra_error']))*np.pi/180)))
#
## Compute chi-squared array for every possible cp position
#for i in list(range(0,len(ra_cp)+1)):
#    for j in list(range(len(dec_cp)+1)):
#        theta = np.arctan(np.sin((ra_cp[i] - table_data[ra_key])*np.pi/180)/(np.cos(table_data[dec_key]*np.pi/180)*np.tan(dec_cp[j]*np.pi/180)-np.sin(table_data[dec_key]*np.pi/180)*np.cos((ra_cp[i]-table_data[ra_key])*np.pi/180)))
#        
#        mu_ll = np.sin(theta)*table_data[pmra_key] + np.cos(theta)*table_data[pmdec_key]
#        mu_t = -np.cos(theta)*table_data[pmra_key] + np.sin(theta)*table_data[pmdec_key]
#        sigma_t_squared = (sigma_theta*mu_ll)**2 + (table_data['pmra_error']*np.cos(theta))**2 + (table_data['pmdec_error']*np.sin(theta))**2
#        
#        t_t_squared = mu_t**2/sigma_t_squared
#
#        t_t_squared_array[i][j] = t_t_squared.sum()
# 
## Determines true cp (which minimises chi-squared)       
#cp_pos = np.unravel_index(t_t_squared_array.argmin(), t_t_squared_array.shape)
#cp_ra = cp_pos[0]
#cp_dec = cp_pos[1]-90
#
## Determines true t-tangent
#theta = np.arctan(np.sin((cp_ra - table_data[ra_key])*np.pi/180)/(np.cos(table_data[dec_key]*np.pi/180)*np.tan(cp_dec*np.pi/180)-np.sin(table_data[dec_key]*np.pi/180)*np.cos((cp_ra-table_data[ra_key])*np.pi/180)))
#       
#mu_ll = np.sin(theta)*table_data[pmra_key] + np.cos(theta)*table_data[pmdec_key]
#mu_t = -np.cos(theta)*table_data[pmra_key] + np.sin(theta)*table_data[pmdec_key]
#sigma_t_squared = (sigma_theta*mu_ll)**2 + (table_data['pmra_error']*np.cos(theta))**2 + (table_data['pmdec_error']*np.sin(theta))**2
#        
#true_t_t_squared = mu_t**2/sigma_t_squared
#
## Calculates probability for each star belonging
#p_cp = np.exp(-0.5*true_t_t_squared) 
#p_cp.name = 'p_cp'

#cp_ra, cp_dec, reduced_ra, reduced_dec, p_cp = convergent_point('Pleiades_data_GaiaDR2')
#cp_ra, cp_dec, reduced_ra, reduced_dec, p_cp = convergent_point('Reduced_OB2_with_RV')

################## GALACTIC XYZ UVW SPACE (Heliocentric) ######################
# Now in Galactic XYZ UVW space:

# Calculate  current XYZ Galactic positions (X = d*cos(b)*cos(l); Y = d*cos(b)*sin(l); Z = d*sin(b) --- (pc)
x_g = np.array([table_data[d_key][i] * cos(table_data['b'][i]*np.pi/180) * cos(table_data['l'][i]*np.pi/180) for i,d1 in enumerate(table_data['b'])]) 
y_g = np.array([table_data[d_key][i] * cos(table_data['b'][i]*np.pi/180) * sin(table_data['l'][i]*np.pi/180) for i,d1 in enumerate(table_data['b'])]) 
z_g = np.array([table_data[d_key][i] * sin(table_data['b'][i]*np.pi/180) for i,d1 in enumerate(table_data['b'])])

# Calculate UVW galactic velocities --- (km/s)
#u_g, v_g, w_g = uvw(ra = table_data['ra_1'], dec = table_data['dec_1'], d = table_data['rest'], pmra = table_data['pmra_1'], pmdec = table_data['pmdec_1'], rv = table_data['radvel']) # OB2
#u_g, v_g, w_g = uvw(ra = table_data[ra_key], dec = table_data[dec_key], d = 1000*1/table_data[plx_key], pmra = table_data[pmra_key], pmdec = table_data[pmdec_key], rv = table_data[rv_key])        # Pleiades
u_g, v_g, w_g = uvw(ra = table_data[ra_key], dec = table_data[dec_key], d = table_data['rest'], pmra = table_data[pmra_key], pmdec = table_data[pmdec_key], rv = table_data[rv_key])        # Hyades


# Calculate old XYZ positions
vel_conv = 1.02269 #((pc/Myr)/(km/s)) --- conversion factor to change velocity units from km/s to pc/Myr 
x_g_old = x_g - u_g*vel_conv*ages
y_g_old = y_g - v_g*vel_conv*ages
z_g_old = z_g - w_g*vel_conv*ages

#Plots XYZ positions at current time
#spatial_plot(x_g, y_g, 1, 'Galactic XY Position of OB2 members \n Current time', 'X (pc)', 'Y (pc)')
#spatial_plot(x_g, z_g, 1, 'Galactic XZ Position of OB2 members \n Current time', 'X (pc)', 'Z (pc)')
#spatial_plot(y_g, z_g, 1, 'Galactic YZ Position of OB2 members \n Current time', 'Y (pc)', 'Z (pc)')

##Plots positions of association at 'age' of association
#spatial_plot(x_g_old, y_g_old, 1, 'Galactic XY Position of OB2 members \n {0} Myr ago'.format(table_data['age'][0]), 'X (pc)', 'Y (pc)')
#spatial_plot(x_g_old, z_g_old, 1, 'Galactic XZ Position of OB2 members \n {0} Myr ago'.format(table_data['age'][0]), 'X (pc)', 'Z (pc)')
#spatial_plot(y_g_old, z_g_old, 1, 'Galactic YZ Position of OB2 members \n {0} Myr ago'.format(table_data['age'][0]), 'Y (pc)', 'Z (pc)')

##Plot position at current time with arrows for velocities
spatial_plot_with_arrows(y_g, x_g, v_g, u_g, 10, 'Galactic XY Position of Hyades members, with overplotted UV velocities \n Current time', 'Y (pc)', 'X (pc)')
spatial_plot_with_arrows(x_g, z_g, u_g, w_g, 10, 'Galactic XZ Position of Hyades members, with overplotted UW velocities \n Current time', 'X (pc)', 'Z (pc)')
spatial_plot_with_arrows(y_g, z_g, v_g, w_g, 10, 'Galactic YZ Position of Hyades members, with overplotted VW velocities \n Current time', 'Y (pc)', 'Z (pc)')

################# GALACTIC XYZ UVW SPACE (Galactocentric) #####################
c1 = coord.ICRS(ra =table_data[ra_key], dec = table_data[dec_key], distance = table_data[d_key], pm_ra_cosdec = table_data[pmra_key], pm_dec = table_data[pmdec_key], radial_velocity = table_data[rv_key])
gc1 = c1.transform_to(coord.Galactocentric)

#spatial_plot(gc1.x, gc1.y, 1, 'Galactocentric XY Position of confirmed Hyades members \n Current time', 'X (pc)', 'Y (pc)')
#spatial_plot_with_arrows(gc1.x, gc1.y, gc1.v_x, gc1.v_y, 200, 'Galactocentric XY Position of confirmed Hyades members, with overplotted UV velocities \n Current time', 'Y (pc)', 'X (pc)')

########################## MAKING ANIMATIONS ##################################

#Set up figure, axis and plot element
# XY
#fig = plt.figure()
#ax = plt.axes(xlim = (0,500), ylim = (-150,400))
#particles, = ax.plot([], [], 'bo', ms=1)
#time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#plt.title('Animation of Galactic XY Position for confirmed OB2 Association members \n t = 0 to 17Myr ago')
#plt.xlabel('X (pc)')
#plt.ylabel('Y (pc)')
#
#def init():
#    particles.set_data([], [])
#    time_text.set_text('')
#    return particles, time_text
#
#def animate(i):
#    time = np.linspace(0, 17, 1000) #Myrs
#    x_g_old = x_g - u_g*vel_conv*time[i]
#    y_g_old = y_g - v_g*vel_conv*time[i]
#    particles.set_data(x_g_old,y_g_old)
#    time_text.set_text('Time = - %.1f Myr' % time[i])
#    return particles,time_text
#
#anim = animation.FuncAnimation(fig, animate, init_func = init, frames = 1000, interval = 20, blit = True)
#
#anim.save('xy_time_evolution.mp4', fps = 60, extra_args = ['-vcodec', 'libx264'])
#
#plt.show()
#
## XZ
#fig2 = plt.figure()
#ax2 = plt.axes(xlim = (0,500), ylim = (-20,300))
#particles2, = ax2.plot([], [], 'bo', ms=1)
#time_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
#plt.title('Animation of Galactic XZ Position for confirmed OB2 Association members \n t = 0 to 17Myr ago')
#plt.xlabel('X (pc)')
#plt.ylabel('Z (pc)')
#
#def init2():
#    particles2.set_data([], [])
#    time_text2.set_text('')
#    return particles2, time_text2
#
#def animate2(i):
#    time = np.linspace(0, 17, 1000) #Myrs
#    x_g_old = x_g - u_g*vel_conv*time[i]
#    z_g_old = z_g - w_g*vel_conv*time[i]
#    particles2.set_data(x_g_old,z_g_old)
#    time_text2.set_text('Time = - %.1f Myr' % time[i])
#    return particles2,time_text2
#
#anim2 = animation.FuncAnimation(fig2, animate2, init_func = init2, frames = 1000, interval = 20, blit = True)
#
#anim2.save('xz_time_evolution.mp4', fps = 60, extra_args = ['-vcodec', 'libx264'])
#
#plt.show()
#
## YZ
#fig3 = plt.figure()
#ax3 = plt.axes(xlim = (-150,600), ylim = (-20,300))
#particles3, = ax3.plot([], [], 'bo', ms=1)
#time_text3 = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)
#plt.title('Animation of Galactic YZ Position for confirmed OB2 Association members \n t = 0 to 17Myr ago')
#plt.xlabel('Y (pc)')
#plt.ylabel('Z (pc)')
#
#def init3():
#    particles3.set_data([], [])
#    time_text3.set_text('')
#    return particles3, time_text3
#
#def animate3(i):
#    time = np.linspace(0, 17, 1000) #Myrs
#    y_g_old = y_g - v_g*vel_conv*time[i]
#    z_g_old = z_g - w_g*vel_conv*time[i]
#    particles3.set_data(y_g_old,z_g_old)
#    time_text3.set_text('Time = - %.1f Myr' % time[i])
#    return particles3,time_text3
#
#anim3 = animation.FuncAnimation(fig3, animate3, init_func = init3, frames = 1000, interval = 20, blit = True)
#
#anim3.save('yz_time_evolution.mp4', fps = 60, extra_args = ['-vcodec', 'libx264'])
#
#plt.show()


###############################################################################

stop = timeit.default_timer()

print('Time: ',stop - start)