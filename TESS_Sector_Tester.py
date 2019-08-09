#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:41:43 2019

TESS Sector tester

This script takes a votable full of targets in the sky and checks when/if TESS 
will observe them. A new row will be added to the end of the table describing
whether it will be viewed, and if so, in which sectors.

@author: phrhzn
"""

import astropy.table as tab
import csv
import pickle
#import numpy as np
#import matplotlib.pyplot as plt
#import timeit

# Read data from table
Table = tab.Table
#table_data = Table.read('BANYAN_XI-III_combined_members_TESS-sectors.xml')
table_data = Table.read("BANYAN_XI-III_members_with_TIC.csv" , format='ascii.csv')

# Build table of targets for each sector
table_data['sector_list'] = [None]*len(table_data['main_id'])

sector_array = ['S1',   'S2',  'S3',  'S4',  'S5',  'S6',  'S7',  'S8',  'S9', 
                'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 
                'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26']

target_dict = {}

for i in sector_array:
    target_dict[i] = []
    for j, data in enumerate(table_data[i]):
        if data != 0:
            target_dict[i].append(table_data['main_id'][j])

with open('TESS_YSO_sector_dict2.pkl', 'wb') as f:
    pickle.dump(target_dict, f, pickle.HIGHEST_PROTOCOL)
 
# Save dictionary to csv
#w = csv.writer(open("TESS_sector_targets.csv", "w"))
#for key, val in target_dict.items():
#    w.writerow([key, val])
    
sector3_targets = target_dict['S3']

with open('Sector_3_targets_from_TIC_list.pkl', 'wb') as f:
    pickle.dump(sector3_targets, f, pickle.HIGHEST_PROTOCOL)