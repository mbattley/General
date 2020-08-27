#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:37:45 2020

eyeball_from_file.py
--------------------
This script allows you to replot any of the eyeballing plots from scratch using 
saved data for original, deterended, lowess-fitted lcs and periododram period/powers


@author: mbattley
"""

from lc_download_methods_new import lc_from_csv

save_path = ''
tic = ''

# Open detrended data for lightcurve
detrended_lc = lc_from_csv(save_path + 'Detrended_lcs/{}_detrended_lc.csv'.format(tic))
time = detrended_lc.time
detrended_flux = detrended_lc.flux