# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:36:48 2019

This short script can be used to convert standard csv files to astropy tables

@author: MatthewTemp
"""
from astropy.table import Table
from astropy.io.votable import from_table, writeto

#filename = "BANYAN_XI-III_members_with_TIC.csv"
#filename = "/Users/mbattley/Documents/PhD/Papers/Detrending of young stars - TESS 1-5/Paper corrections/Grid_sensitivity_analysis/Full Sensitivity Analysis Table.csv"
filename = '/Users/mbattley/Documents/PhD/young_star_lists/Final_young_star_list_MB_EG_20201228_TICv8_matched_unordered.csv'

def csv_to_votable(filename, save_filename):
    data = Table.read(filename, format='ascii.csv')

    votable = from_table(data)

    writeto(votable, save_filename)

data = Table.read(filename, format='ascii.csv')

votable = from_table(data)

writeto(votable, "/Users/mbattley/Documents/PhD/Young_Star_Lists/Final_young_star_list_MB_EG_20201228_TICv8_matched_unordered.xml")