#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:29:24 2020

Convert textfile to votable.

@author: mbattley
"""

from astropy.table import Table
from astropy.io.votable import from_table, writeto
import numpy as np
from astropy.io import ascii
import pandas as pd

save_path = "/Users/mbattley/Documents/PhD/Young Star Lists/"

#filename = "BANYAN_XI-III_members_with_TIC.csv"
filename = save_path+"Bell17_simplified.txt"


#data - pd.read_table(filename)
data = Table.read(filename, format='ascii')
#data = np.fromfile(filename, dtype=float)

votable = from_table(data)

writeto(votable, save_path + "Bell17.xml")