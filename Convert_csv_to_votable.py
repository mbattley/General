# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:36:48 2019

This short script can be used to convert standard csv files to astropy tables

@author: MatthewTemp
"""
from astropy.table import Table
from astropy.io.votable import from_table, writeto

filename = "TESS_sector1.csv"

data = Table.read(filename, format='ascii.csv')

votable = from_table(data)

writeto(votable, "TESS_sector_1_targets.xml")