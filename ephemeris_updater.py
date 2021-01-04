#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:31:05 2020
ephemeris_updater.py

This script build new ephemerides from individual transit times and constructs 
O-C diagrams

@author: mpbattley
"""

import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(n,per,t0):
    return n*per + t0

def quad_func(n,a,b,c):
    return a*n**2 + b*n + c

def tri_func(n,a,b,c,d):
    return a*n**3 + b*n**2 + c*n + d

def sine_func(n,a,b,c,d,e,f):
    return a*np.sin(b*n) + c

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(list_of_elem)


save_path = '/Users/mbattley/Documents/PhD/Kepler-2min xmatch/'
Kepler_name = 'Kepler-411'
planet_letter = 'd'
plot_epochs = True
filename = save_path + 'Final_Ephemerides.csv'

#planet_data = Table.read(save_path + 'Kepler_planets_reobserved_in_TESS_2min.csv', format='ascii.csv')
#planet_data = Table.read(save_path + 'Kepler_pcs_reobserved_in_TESS_2min_final.csv', format='ascii.csv')

with open(save_path + 'Final t0s/Overall t0s for {} {}.csv'.format(Kepler_name, planet_letter), 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    t0s = np.array(next(csv_reader))
    sds = np.array(next(csv_reader))
    t0s = np.array([float(d) for d in t0s])
    sds = np.array([float(d) for d in sds])

mask = np.ones(len(t0s), dtype=bool)
#dodgy_times_HAT_P_7 = [-5,-11,4,59,150,178,246,360,374,427,429,466,480,500,557,576]
#dodgy_times_KOI_13b = [0,92,139,205,213,245,251,285,306,318,508,528,577,616]
#dodgy_times = [6]
#for i in dodgy_times:
#    mask[i] = False
#t0s = t0s[mask]
#sds = sds[mask]

period = t0s[1] - t0s[0]
transit_ns = np.array([round((item-t0s[0])/period) for item in t0s])

#combined_period, combined_fit_t0 = np.polyfit(transit_ns, t0s,1)
#y_model_overall = np.array(transit_ns)*combined_period + combined_fit_t0

# Normal linear fit, using curve_fit for weighted fitting
popt, pcov = curve_fit(func, transit_ns, t0s, sigma=sds, absolute_sigma=True)
y_model_overall = func(transit_ns, *popt)
overall_fit_period1 = popt[0]
overall_fit_t01 = popt[1]
perr = np.sqrt(np.diag(pcov))

#Second iteration:
transit_ns = np.array([round((item-overall_fit_t01)/overall_fit_period1) for item in t0s])

popt, pcov = curve_fit(func, transit_ns, t0s, sigma=sds, absolute_sigma=True)
y_model_overall = func(transit_ns, *popt)
overall_fit_period2 = popt[0]
overall_fit_t02 = popt[1]
perr = np.sqrt(np.diag(pcov))

#Third iteration:
transit_ns = np.array([round((item-overall_fit_t02)/overall_fit_period2) for item in t0s])

popt, pcov = curve_fit(func, transit_ns, t0s, sigma=sds, absolute_sigma=True)
y_model_overall = func(transit_ns, *popt)
overall_fit_period = popt[0]
overall_fit_t0 = popt[1]
perr = np.sqrt(np.diag(pcov))

# Test quadratic one in case
#popt, pcov = curve_fit(quad_func, transit_ns, t0s, sigma=sds, absolute_sigma=True)
#y_model_overall = quad_func(transit_ns, *popt)
##
# Test 3rd order one in case
#popt, pcov = curve_fit(tri_func, transit_ns, t0s, sigma=sds, absolute_sigma=True)
#y_model_overall = tri_func(transit_ns, *popt)

# Test other weird ones
#popt, pcov = curve_fit(sine_func, transit_ns, t0s, sigma=sds, absolute_sigma=True)
#y_model_overall = sine_func(transit_ns, *popt)


plt.figure()
plt.errorbar(transit_ns, t0s, yerr=sds, fmt='.', c='k')
plt.plot(transit_ns,y_model_overall)
plt.xlabel('Transits since T0')
plt.ylabel('BJD Time [Days]')

# Calculate overall O-C and replot entire O-C diagram
o_c_combined = t0s - y_model_overall
o_c_hrs_combined = o_c_combined*24
e_overall_hrs = sds*24

o_c_plot_final = plt.figure()
plt.errorbar(t0s, o_c_hrs_combined, yerr=e_overall_hrs, fmt='.', c='k')
plt.xlabel('BJD Time [Days]')
plt.ylabel('O-C [hrs]')
#plt.title('O-C diagram for {} {} including TESS'.format(Kepler_name, planet_letter))
plt.show()

o_c_plot_min = plt.figure()
plt.errorbar(t0s, o_c_hrs_combined*60, yerr=e_overall_hrs*60, fmt='.', c='k')
plt.xlabel('BJD Time [Days]')
plt.ylabel('O-C [mins]')
plt.title('O-C diagram for {} {} including TESS'.format(Kepler_name, planet_letter))
plt.show()

# Optional: save directly
#append_list_as_row(filename,[Kepler_name+planet_letter,popt[0],perr[0],popt[1],perr[1]])

if plot_epochs == True:
    with open(Kepler_name + '_data.pkl', 'rb') as f:
        kepler_data_reopened = pickle.load(f)
        time = kepler_data_reopened['time_combined'] 
        flux = kepler_data_reopened['flux_combined'] +1
        time_TESS = kepler_data_reopened['time_TESS'] 
        flux_TESS = kepler_data_reopened['flux_TESS'] +1

    plt.figure()
    plt.scatter(time, flux, c='k', s=1, label="Original Data")
    for t0i in t0s:
        plt.axvline(x=t0i, c='r')
    plt.legend()
    plt.title('{} {} initial and final T0s after one run'.format(Kepler_name, planet_letter))
