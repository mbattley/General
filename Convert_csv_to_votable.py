# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:36:48 2019

This short script can be used to convert standard csv files to astropy tables

@author: MatthewTemp
"""
from astropy.table import Table
from astropy.io.votable import from_table, writeto

filename = "BANYAN_XI-III_combined_members.csv"

data = Table.read(filename, format='ascii.csv')

votable = from_table(data)

writeto(votable, "BANYAN_XI-III_combined_members_TESS-sectors.xml")