#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:32:12 2019
utility_belt.py

This script collects a series of functions that are useful oevr many different
codes

@author: mbattley
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astroquery.mast import Catalogs

def tic_stellar_info(target_ID, from_file = False, filename = 'BANYAN_XI-III_members_with_TIC.csv'):
    if from_file == True:
        table_data = Table.read(filename , format='ascii.csv')
        
        # Obtains ra and dec for object from target_ID
        i = list(table_data['main_id']).index(target_ID)
        #camera = table_data['S{}'.format(sector)][i]
        tic = table_data['MatchID'][i]
        r_star = table_data['Stellar Radius'][i]
        T_eff = table_data['T_eff'][i]
        
    else:
        TIC_table = Catalogs.query_object(target_ID, catalog = "TIC")
        tic = TIC_table['ID'][0]
#        print(TIC_table[0])
        r_star = TIC_table['rad'][0]
        T_eff = TIC_table['Teff'][0]
    
    return tic, r_star, T_eff

def planet_size_from_depth(target_ID, depth):
    tic, r_star, T_eff = tic_stellar_info(target_ID)
    
    r_Sun = 695510 #km
    r_Jup = 69911  #km
    r_Nep = 24622  #km
    r_Earth = 6371 #km
    
    r_p_solar = r_star*np.sqrt(depth) # Radius in Solar radii
    r_p_km = r_p_solar*r_Sun          # Radius in km
    r_p_Jup = r_p_km/r_Jup            # Radius in Jupiter radii
    r_p_Nep = r_p_km/r_Nep            # Radius in Neptune radii
    r_p_Earth = r_p_km/r_Earth        # Radius in Earth radii
    
    print('Planet size:')
    print('{} R_solar'.format(r_p_solar))
    print('{}km'.format(r_p_km))
    print('{} R_Jup'.format(r_p_Jup))
    print('{} R_Nep'.format(r_p_Nep))
    print('{} R_Earth'.format(r_p_Earth))
    
    return r_p_Jup
        