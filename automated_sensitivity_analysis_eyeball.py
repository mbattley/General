#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:34:48 2020

Script to more automatically analyse sensitivity analysis grid results

@author: mbattley
"""
import numpy as np
import astropy
from astropy.table import Table

save_path = '/Users/mbattley/Documents/PhD/Papers/Detrending of young stars - TESS 1-5/Paper corrections/Grid2_sensitivity/'
sector = 1

sector_list = [1,2,3,4,5]

for sector in sector_list:
    table_data = Table.read(save_path + "Sensitivity_analysis_S{}_grid_cleaned.csv".format(sector) , format='ascii.csv', guess = False)
    
    table_data['Recovered'] = 0
    
    for i in range(len(table_data['Target ID'])):
        if np.abs(table_data['Max Period'][i] - table_data['Injected Period'][i]) < 0.02:
            table_data['Recovered'][i] = table_data['Injected Period'][i]
        elif np.abs(table_data['Max Period'][i]-table_data['Injected Period'][i]*2) <0.02:
            table_data['Recovered'][i] = table_data['Injected Period'][i]
        elif np.abs(table_data['Max Period'][i]-table_data['Injected Period'][i]*0.5) <0.02:
            table_data['Recovered'][i] = table_data['Injected Period'][i]
        elif np.abs(table_data['2nd highest period'][i]-table_data['Injected Period'][i]) <0.02:
            table_data['Recovered'][i] = table_data['Injected Period'][i]
        elif np.abs(table_data['3rd Highest Period'][i]-table_data['Injected Period'][i]) <0.02:
            table_data['Recovered'][i] = table_data['Injected Period'][i]
        elif table_data['Injected Period'][i] == 14:
            if np.abs(table_data['Max Period'][i] - table_data['Injected Period'][i]) < 0.2:
                table_data['Recovered'][i] = table_data['Injected Period'][i]
        
        # Fixes J0638-5604 due to inconvenient rotation period
        if table_data['Target ID'][i] == 'J0638-5604':
            table_data['Recovered'][i] = 0
        
            
    #print(table_data['Recovered'])
    
    astropy.io.ascii.write(table_data, save_path+'Sensitivity_analysis_S{}_grid_final.csv'.format(sector), format='csv',overwrite = True) 