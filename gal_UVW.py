#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 08:55:12 2018

Script containing a function to calculate galactic UVW velocities given ra, dec, distance, pmra, pmdec and RV
Adapted from NASA's IDL code to do the same: http://idlastro.gsfc.nasa.gov/ftp/pro/astro/gal_uvw.pro
n.b. this is in turn based on the method from Johnson D.R.H & Soderblom D.R., 1987, AJ, 93, pp 864-867

@author: Matthew Battley
"""

import numpy as np
import astropy.table as tab
from math import sin, cos

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

# Read data from table
Table = tab.Table
uvw_data = Table.read('OB2_Gaia_Zeeuw_Match_dist')

# Change from unrecognisable unit names in file
uvw_data['pmra'].unit = 'mas/yr'
uvw_data['pmdec'].unit = 'mas/yr'
uvw_data['ra'].unit = 'deg'
uvw_data['dec'].unit = 'deg'

# Calculate  current XYZ Galactic positions (X = d*cos(b)*cos(l); Y = d*cos(b)*sin(l); Z = d*sin(b) --- (pc)
x_g = np.array([uvw_data['rest'][i] * cos(uvw_data['b'][i]*np.pi/180) * cos(uvw_data['l'][i]*np.pi/180) for i,d1 in enumerate(uvw_data['b'])]) 
y_g = np.array([uvw_data['rest'][i] * cos(uvw_data['b'][i]*np.pi/180) * sin(uvw_data['l'][i]*np.pi/180) for i,d1 in enumerate(uvw_data['b'])]) 
z_g = np.array([uvw_data['rest'][i] * sin(uvw_data['b'][i]*np.pi/180) for i,d1 in enumerate(uvw_data['b'])])

# Calculate UVW galactic velocities --- (km/s)
u_g, v_g, w_g = uvw(ra = uvw_data['ra'], dec = uvw_data['dec'], d = uvw_data['rest'], pmra = uvw_data['pmra'], pmdec = uvw_data['pmdec'], rv = uvw_data['radial_velocity'])
