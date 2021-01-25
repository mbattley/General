#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:10:09 2021

@author: mbattley
"""

import pandas as pd
import numpy as np

data = pd.read_csv('/Users/mbattley/Documents/PhD/young_star_lists/Final_young_star_list_MB_EG_20201228_TICv8_matched_unordered.csv')

data.to_csv("/Users/mbattley/Documents/PhD/Young_Star_Lists/Final_young_star_list_MB_EG_20201228_TICv8_matched_unordered2.csv")

np.where(pd.isnull(data))