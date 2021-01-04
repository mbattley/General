#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:52:52 2020

TTV analysis
------------

Main aim: to create O-C analysis for a given lightcurve

@author: mbattley
"""


import sys
import corner
import pickle
import time as timing
import batman
import scipy
import csv
import json
import lightkurve
import numpy as np
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
import statsmodels.api as sm
from astropy.table import Table
from lc_download_methods import two_min_lc_download
from remove_tess_systematics import clean_tess_lc
from astropy.io import fits
from astropy.time import Time
from astropy import coordinates as coord
from astropy import constants as const
from exoplanet.gp import terms
from lightkurve import search_lightcurvefile
from scipy.signal import find_peaks
from utility_belt import binned
from scipy import interpolate
from lowess_detrend import lowess_detrending

plt.rcParams.update({'figure.max_open_warning': 0})

# n.b. steps for fixing new systems:
# 1. Check whether the flux is set at 0 or 1
# 2. Check epochs are correct

def mask_planet(time,flux,epoch,period,duration,target_ID,pl_letter):
    phase = np.mod(time-epoch-period/2,period)/period
        
    plt.figure()
    plt.scatter(phase, flux, c= 'k', s=2)
    plt.title('{} data folded by planet {} period'.format(target_ID, pl_letter))
    
    near_transit = [False]*len(flux)
    
    if pl_letter == 'c':
        for i in range(len(time)):
            if abs(phase[i] - 0.5) < duration/period:
                near_transit[i] = True
    else:
        for i in range(len(time)):
            if abs(phase[i] - 0.5) < duration/period:
                near_transit[i] = True
    
    near_transit = np.array(near_transit)
    
    time_masked = time[~near_transit]
    flux_masked = flux[~near_transit]
    phase_masked = phase[~near_transit]

    plt.figure()
    plt.scatter(phase_masked, flux_masked, c='k', s=2)
    plt.title('Kepler data after planet {} masked'.format(pl_letter))
    
    return time_masked, flux_masked


def find_ttvs(initial_t0is, period, time, flux, params, search_width, step_size,run_number=2):
    n_list = range(len(initial_t0is))

    final_t0is = [1]*len(n_list)
    final_t0i_cs = [1]*len(n_list)
    
    num_transits_done = 0
    
    for n in n_list:
        int_start = initial_t0is[n] - 2
        int_end = initial_t0is[n] + 2
        idx = np.where((time > int_start) & (time< int_end))
        
        int_time = time[idx]
        int_flux = flux[idx]
        
        t0i_list = np.arange(initial_t0is[n]-search_width,initial_t0is[n]+search_width,step_size)
        
#        chi_sq_list = []   
        
        for i in range(len(t0i_list)):
            params.t0 = t0i_list[i]                       #time of inferior conjunction
        
            m = batman.TransitModel(params, int_time)
            calc_flux = m.light_curve(params) 
            
            chi_sq = scipy.stats.chisquare(int_flux, f_exp=calc_flux)
    #        chi_sq_list.append((chi_sq[0],params.t0))
            
    #        plt.figure()
    #        plt.scatter(int_time, int_flux, c='k', s=2)
    #        plt.plot(int_time, calc_flux)
    #        txt = "Chi-sq = {}; p-value = {}".format(chi_sq[0], chi_sq[1])
    #        plt.annotate(
    #                txt,
    #                (0, 0),
    #                xycoords="axes fraction",
    #                xytext=(5, 5),
    #                textcoords="offset points",
    #                ha="left",
    #                va="bottom",
    #                fontsize=12,
    #        )
            if chi_sq[0] < final_t0i_cs[n]:
                final_t0is[n] = params.t0
                final_t0i_cs[n] = chi_sq[0]
            print('Number of transits analysed (round {}) = {}'.format(run_number,num_transits_done))
            num_transits_done += 1
    
    final_t0is = np.array(final_t0is)
    initial_t0is = np.array(initial_t0is)
    
    o_c = final_t0is - initial_t0is

    return final_t0is, o_c

#Psudo-code:
# Pull in random light-curve of system with known planet
# Detrend out of transit effects - probably with Wotan, polynomial or long-window lowess
#       -> Possibly do this transit by transit like previous papers?
# Access priors for planetary system, most importantly period and ephemeris
# Build initial batman light-curve from this (or from stacked one)
# Fitting each individual transit, using either of:
#    - Move along light-curve bit by bit according to known period and ephemeris
#    - Cut lightcurve section about one duration either side of each transit and fit each of these
#       -> allow only transit time to vary...
# Note down observed (lowest chi-squared over a range of times) value for each transit
# Conctruct O-C diagram by comparing observed ones from last step to original from archival period/ephermeris
# Use O-C ones to create stacked model, model with exoplanet (or similar?) and redo TTV analysis with 
# Iterate if necessary, rebuilding new model (esp. for those with largest TTVs)

# Idea: could use find-peaks function as first guess to get well-folded one
start = timing.time()
############################## PART 0: Setup ##################################
# lc parameters
save_path = '/Users/mbattley/Documents/PhD/Kepler-2min xmatch/'
target_ID = 'TIC 417676622' # Kepler 25 test-case: 'TIC 120960812' #TIC number
Kepler_name = 'Kepler-68'
planet_letter = 'b'
TIC = int(target_ID[4:])
sector = 14
multi_sector = False
planet_data = Table.read(save_path + 'Kepler_planets_reobserved_in_TESS_2min.csv', format='ascii.csv')
#planet_data = Table.read(save_path + 'Kepler_pcs_reobserved_in_TESS_2min_final.csv', format='ascii.csv')

#target_ID_list = np.array(pc_data['TICID'])
i = list(planet_data['TICID']).index(int(target_ID[3:]))
instrument = 'both'
planet_letters = ['b','c','d','e','f','g','h']
method = 'auto' #Can be 'array, 'auto' or 'manual'
pc = False
transit_mask =False
transit_mask_TESS = False
transit_cut = False
user_defined_masking = True
detrending = 'lowess_full'
large_TTV = False
small_TTV = False
no_TTV = True
if planet_letter == 'b':
    planet_num = 0
elif planet_letter == 'c':
    planet_num = 1
elif planet_letter == 'd':
    planet_num = 2
#
##################### PART 1: Downloading/Opening Light-curves ##################
#TESS 2min
#if (planet_data['S14'][i] != 0) and (planet_data['S15'][i] != 0) and (planet_data['S26'][i] != 0):
#    multi_sector = [14,15,26]
##    if (planet_data['S26'][i] != 0):
##        multi_sector = [14,15]
#elif (planet_data['S14'][i] != 0) and (planet_data['S15'][i] != 0):
#    multi_sector = [14,15]
#elif (planet_data['S14'][i] != 0) and (planet_data['S26'][i] != 0):
#    multi_sector = [14,26]
#
##multi_sector = False
#if multi_sector != False:
#    sap_lc, pdcsap_lc = two_min_lc_download(TIC, sector = multi_sector[0], from_file = False)
#    lc = pdcsap_lc
#    nancut = np.isnan(lc.flux) | np.isnan(lc.time)
#    lc = lc[~nancut]
#    for sector_num in multi_sector[1:]:
#        sap_lc_new, pdcsap_lc_new = two_min_lc_download(TIC, sector_num, from_file = False)
#        lc_new = pdcsap_lc_new
#        nancut = np.isnan(lc_new.flux) | np.isnan(lc_new.time)
#        lc_new = lc_new[~nancut]
#        lc = lc.append(lc_new)
#else:
##    sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = sector, from_file = False)
#    lcf = search_lightcurvefile(target_ID, sector=sector).download()
#    pdcsap_lc = lcf.PDCSAP_FLUX
#    header_0 = lcf
#    lc = pdcsap_lc
#    nancut = np.isnan(lc.flux) | np.isnan(lc.time)
#    lc = lc[~nancut]
#    print('Removed nans')
#time_TESS = np.array(lc.time) #n.b. in TJD (TESS Time)
#time_TESS_orig = np.array([float(str(element).strip()) for element in time_TESS]) + 2457000 #Convert to BJD for consistency
#flux_TESS = lc.flux
#flux_TESS_orig = np.array(flux_TESS)/np.median(flux_TESS) -1 #Normalizes and sets mean to zero, as in exoplanet tutorial
#flux_err_TESS = lc.flux_err/np.median(flux_TESS)
#mean_flux_err_TESS = np.mean(flux_err_TESS)
#
## Kepler
##KIC = planet_data['KIC'][i]
#KIC = planet_data['kepid'][i]
#lcfs = lightkurve.search_lightcurvefile(KIC, mission='Kepler').download_all()
#stitched_lc = lcfs.PDCSAP_FLUX.stitch()
##stitched_lc = lcfs.PDCSAP_FLUX.stitch(corrector_func=my_custom_corrector_func)
#nancut = np.isnan(stitched_lc.flux) | np.isnan(stitched_lc.time)
#stitched_lc = stitched_lc[~nancut]
#time_Kepler = np.array(stitched_lc.time) + 2454833 #Convert to BJD for consistency
#flux_Kepler = np.array(stitched_lc.flux)/np.median(stitched_lc.flux) -1
#binned_Kepler_time, binned_Kepler_flux = binned(time_Kepler,flux_Kepler,binsize=2)
#yerr = stitched_lc.flux_err
#
#if instrument == 'TESS':
#    time_TESS = time_TESS_orig
#    flux_TESS = flux_TESS_orig
#elif instrument == 'Kepler':
#    time_TESS = time_Kepler
#    flux_TESS = flux_Kepler
#elif instrument == 'both':
#    time_TESS = np.append(time_Kepler,time_TESS_orig)
#    flux_TESS = np.append(flux_Kepler,flux_TESS_orig)
##
#kep_flux_err = np.mean(stitched_lc.flux_err)
#kepler_data = {'time_combined':time_TESS, 'flux_combined':flux_TESS, 'time_TESS':time_TESS_orig, 'flux_TESS':flux_TESS_orig, 'time_Kepler':time_Kepler, 'flux_Kepler':flux_Kepler, 'flux_err':kep_flux_err, 'flux_err_TESS':mean_flux_err_TESS }
#with open(Kepler_name + '_data.pkl', 'wb') as f:
#    pickle.dump(kepler_data,f)                  

with open(Kepler_name + '_data.pkl', 'rb') as f:
    kepler_data_reopened = pickle.load(f)
    time = kepler_data_reopened['time_combined'] 
    flux = kepler_data_reopened['flux_combined'] +1
    time_TESS = kepler_data_reopened['time_TESS'] 
    flux_TESS = kepler_data_reopened['flux_TESS'] +1
    time_Kepler = kepler_data_reopened['time_Kepler']
    flux_Kepler = kepler_data_reopened['flux_Kepler'] +1
    flux_err = kepler_data_reopened['flux_err']
    mean_flux_err_TESS = kepler_data_reopened['flux_err_TESS']

time = time_Kepler - 2454833
flux = flux_Kepler
flux_err = 2.0e-4

plt.figure()  
plt.scatter(time,flux, c='k', s=1)
plt.xlabel("Time [BJD - 2454833]")
plt.ylabel("Normalized Flux [ppt]")
plt.show()

#plt.figure()
#plt.scatter(time,flux, c='k', s=1)
#plt.xlabel("Time [BJD]")
#plt.ylabel("Normalized Flux [ppt]")
#plt.show()

##################### PART 1b: Planet Parameters ############################
texp_TESS = 120                      # Kepler (60s)/TESS (120s) exposure time (s)
texp_TESS /= 60.0 * 60.0 * 24.0 	     # converting exposure time to days (important!!!!!!)
texp_Kepler = 30*60
texp_Kepler /= 60.0 * 60.0 * 24.0 

if method == 'manual':
    periodi_b = 6.238297 #From exoplanet archive - make automatic later
    periodi_c = 12.7207  #P in days
    periodi_sd_b = 1.70E-05
    periodi_sd_c = 1.10E-04
    
    t0i_b = 2455703.42			# ephemeris from Exoplanet Archive
    t0i_c = 2455711.15          # BJD. n.b. use - 2454833 for Kepler time/ - 2457000 for TESS
    t0i_sd_b = 0.5					    # stand. dev on the t0
    t0i_sd_c = 0.5
    
    radi_b = 0.245*const.R_jup.value/const.R_sun.value    # Planet radius in solar radii
    radi_c = 0.465*const.R_jup.value/const.R_sun.value
    
    M_star = 1.19, 0.06		# mass, uncertainty (solar masses)
    R_star = 1.31, 0.02	    # radius, uncertainty (solar radii)
elif pc == True:
    num_planets =0
    planet_params = Table(names=('periodi','periodi_sd','t0i','t0i_sd','t0i_sd_real','radi','a','incl','ecc','eccs_sd'))
    for j in range(len(planet_data['TICID'])):
        if planet_data['TICID'][j] == TIC:
            periodi = planet_data['koi_period'][j]
            period_sd = planet_data['koi_period_err1'][j]
            t0i = planet_data['koi_time0'][j]
            t0i_sd_real = planet_data['koi_time0_err1'][j]
            t0i_sd = 0.1
            M_star = planet_data['koi_smass'][j],planet_data['koi_smass_err1'][j]
            # Gets updated R_star based on DR2 (Berger+16) if available, otherwise falls back to exo archive
            if ma.is_masked(planet_data['R*'][j]) == False:
#                R_star = planet_data['R*'][j], planet_data['E_R*_2'][j]
                R_star = planet_data['koi_srad'][j], planet_data['koi_srad_err1'][j]
            else:
                R_star = planet_data['koi_srad'][j], planet_data['koi_srad_err1'][j]
            radi = planet_data['koi_ror'][j]*R_star[0] # In solar radii
            a = (((const.G.value*M_star[0]*const.M_sun.value*(periodi*86400.)**2)/(4.*(np.pi**2)))**(1./3))/(1.495978707e11)   # in AU
            if ma.is_masked(planet_data['koi_incl'][j]) == False:
#                incl = 83.6
                incl = planet_data['koi_incl'][j]
                if incl > 90:
                    ea_incl = incl
                    incl = 90-(ea_incl - 90)
#                    incl = 87.5
            else:
                incl = 90
            if ma.is_masked(planet_data['koi_eccen'][j]) == False:
                ecc = planet_data['koi_eccen'][j]
                ecc_sd = 0.01
            else: 
                ecc = 0.0
                ecc_sd = 0.01
            planet_params.add_row((periodi, period_sd, t0i, t0i_sd, t0i_sd_real, radi, a, incl, ecc, ecc_sd))
            num_planets += 1
else:
    num_planets =0
    incl = 87.0
    ecc = 0
    planet_params = Table(names=('periodi','periodi_sd','t0i','t0i_sd','t0i_sd_real','radi','a','incl','ecc','eccs_sd'))
    for j in range(len(planet_data['TICID'])):
        if planet_data['TICID'][j] == TIC and planet_data['pl_discmethod'][j] == 'Transit':
            if ma.is_masked(planet_data['Per'][j]) == False:
                periodi = planet_data['Per'][j]
                period_sd = planet_data['e_Per'][j]
                t0i = planet_data['T0'][j]
                t0i_sd_real = planet_data['pl_tranmiderr1'][j]
            else:
                periodi = planet_data['pl_orbper'][j]
                period_sd = planet_data['pl_orbpererr1'][j]
                t0i = planet_data['pl_tranmid'][j]
                t0i_sd_real = planet_data['pl_tranmiderr1'][j]
            t0i_sd = 0.1
#            if planet_num == 1:
#                t0i_sd = 0.05
            if ma.is_masked(planet_data['Rp'][j]) == False:
                radi = planet_data['Rp'][j]*const.R_earth.value/const.R_sun.value # In solar radii
            else:
                radi = planet_data['pl_radj'][j]*const.R_jup.value/const.R_sun.value # In solar radii
            #n.b. if getting it from revised info convert from Earth radii to jupiter radii
            M_star = planet_data['st_mass'][j],planet_data['st_masserr1'][j]
            # Gets updated R_star based on DR2 (Berger+16) if available, otherwise falls back to exo archive
            if ma.is_masked(planet_data['R*'][j]) == False:
                R_star = planet_data['R*'][j], planet_data['E_R*_2'][j]
            else:
                R_star = planet_data['st_rad'][j], planet_data['st_raderr1'][j]
            if ma.is_masked(planet_data['pl_orbsmax'][j]) == False:
                a = planet_data['pl_orbsmax'][j]  #in AU
            else:
                a = (((const.G.value*M_star[0]*const.M_sun.value*(periodi*86400.)**2)/(4.*(np.pi**2)))**(1./3))/(1.495978707e11)   # in AU
            if ma.is_masked(planet_data['pl_orbincl'][j]) == False:
#                incl = 89
                incl = planet_data['pl_orbincl'][j]
                if incl > 90:
                    ea_incl = incl
                    incl = 90-(ea_incl - 90)
#                    incl = 87.5
            else:
                incl = 90
            if ma.is_masked(planet_data['pl_orbeccen'][j]) == False:
                ecc = planet_data['pl_orbeccen'][j]
                ecc_sd = planet_data['pl_orbeccenerr1'][j]
            else: 
                ecc = 0.0
                ecc_sd = 0.01
            planet_params.add_row((periodi, period_sd, t0i, t0i_sd, t0i_sd_real, radi, a, incl, ecc, ecc_sd))
            num_planets += 1
periods = np.array(planet_params['periodi'])
period_sds = np.array(planet_params['periodi_sd'])
t0is = np.array(planet_params['t0i'])
t0i_sds = np.array(planet_params['t0i_sd'])
t0i_sds_real = np.array(planet_params['t0i_sd_real'])
radii = np.array(planet_params['radi'])
a_array = np.array(planet_params['a'])
incls = np.array(planet_params['incl'])
eccs = np.array(planet_params['ecc'])
ecc_sds = np.array(planet_params['eccs_sd'])

# Calculate theoretical transit duration:
#b = a*np.cos(incl)/R_star[0]# Calculate impact parameter
#T_dur = (periods[planet_num]/np.pi)*np.arcsin(np.sqrt((R_star[0]+radii[planet_num])**2-(b*R_star[0])**2)/a)

#################### PART 2: Detrending lightcurve ###########################
##### Transit-masking stuff
#transit_mask=False
if transit_mask == True:
    period = periods[planet_num]
    epoch = t0is[planet_num]
    duration = 0.1 #Or: use calculated duration x 2-4
    phase = np.mod(time_Kepler-epoch-period/2,period)/period
    
    near_transit = [False]*len(flux_Kepler)
    
    for i in range(len(time_Kepler)):
        if abs(phase[i] - 0.5) < duration/period:
            near_transit[i] = True
    
    near_transit = np.array(near_transit)
    
    t_masked = time_Kepler[~near_transit]
    flux_masked = flux_Kepler[~near_transit]
#    flux_err_masked = flux_err_cut[~near_transit]
    t_new = time_Kepler[near_transit]
    pipeline = 'Kepler'
    if pipeline == 'Kepler':
        f = interpolate.interp1d(t_masked,flux_masked, kind = 'slinear')
    else:
        f = interpolate.interp1d(t_masked,flux_masked, kind = 'quadratic')
#                f = interpolate.BarycentricInterpolator(t_masked,flux_masked)

    flux_new = f(t_new)
    interpolated_fig = plt.figure()
    plt.scatter(t_masked, flux_masked, s = 2, c = 'k')
#    plt.scatter(time_Kepler, flux_Kepler, s = 8, c = 'k')
    plt.scatter(t_new,flux_new, s=8, c = 'r')
    plt.xlabel('Time - 2457000 [BTJD days]')
    plt.ylabel('Relative flux')
#    interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
    
    t_transit_mask = np.concatenate((t_masked,t_new), axis = None)
    flux_transit_mask = np.concatenate((flux_masked,flux_new), axis = None)
    
    sorted_order = np.argsort(t_transit_mask)
    t_transit_mask = t_transit_mask[sorted_order]
    flux_transit_mask = flux_transit_mask[sorted_order]


if detrending == 'lowess_full':
    #t_cut = lc_30min.time
    #flux_cut = combined_flux
    full_lowess_flux = np.array([])
    if transit_mask == True:
        lowess = sm.nonparametric.lowess(flux_transit_mask, t_transit_mask, frac=48/len(time_Kepler))
    else:
        lowess = sm.nonparametric.lowess(flux, time, frac=48/len(time_Kepler))
    
    overplotted_lowess_full_fig = plt.figure()
    plt.scatter(time_Kepler,flux_Kepler, c = 'k', s = 2)
#    plt.plot(lowess[:, 0], lowess[:, 1])
    plt.plot(time_Kepler, lowess[:, 1])
    plt.title('{} lc with overplotted lowess full lc detrending'.format(Kepler_name))
    plt.xlabel('Time [BJD]')
    plt.ylabel('Relative flux')
    #overplotted_lowess_full_fig.savefig(save_path + "{} lc with overplotted LOWESS full lc detrending.png".format(target_ID))
    plt.show()
#    plt.close(overplotted_lowess_full_fig)
    
    residual_flux_lowess = flux_Kepler/lowess[:,1]
    full_lowess_flux = np.concatenate((full_lowess_flux,lowess[:,1]))
    
    lowess_full_residuals_fig = plt.figure()
    plt.scatter(time_Kepler,residual_flux_lowess, c = 'k', s = 2)
    plt.title('{} lc after lowess full lc detrending'.format(Kepler_name))
    plt.xlabel('Time [BJD]')
    plt.ylabel('Relative flux')
    ax = plt.gca()
    #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#            lowess_full_residuals_fig.savefig(save_path + "{} lc after LOWESS full lc detrending.png".format(target_ID))
    plt.show()
    
flux_Kepler = residual_flux_lowess

############### PART 2b: Masking other planets and plotting phase ##############
#
## Planet masking
#transit_mask = False
if transit_cut == True:
    if user_defined_masking == True:
        planets_to_mask = [1]
        durations = [0.2,0.2,0.5]
#        durations = [0.25,0.25,0.25]
    elif planet_letter == 'b':
        planets_to_mask = 1
    elif planet_letter == 'c':
        planets_to_mask = 0 #n.b. define planet letter by number - b=0; c=1 etc
    for planet_to_mask in planets_to_mask:
        period = periods[planet_to_mask]
        epoch = t0is[planet_to_mask]
        duration = durations[planet_to_mask] #Adjust accordingly
        time_Kepler, flux_Kepler = mask_planet(time_Kepler,flux_Kepler,epoch,period,duration,Kepler_name,planet_letters[planet_to_mask])
    time_Kepler_masked = time_Kepler
    flux_Kepler_masked = flux_Kepler
#        phase = np.mod(time_Kepler-epoch-period/2,period)/period
#        
#        plt.figure()
#        plt.scatter(phase, flux_Kepler, c= 'k', s=2)
#        plt.title('{} data folded by planet {} period'.format(Kepler_name, planet_letters[planet_to_mask]))
#        
#        near_transit = [False]*len(flux_Kepler)
#        
#        for i in range(len(time_Kepler)):
#            if abs(phase[i] - 0.5) < duration/period:
#                near_transit[i] = True
#        
#        near_transit = np.array(near_transit)
#        
#        time_Kepler_masked = time_Kepler[~near_transit]
#        flux_Kepler_masked = flux_Kepler[~near_transit]
#    
#        plt.figure()
#        plt.scatter(time_Kepler_masked, flux_Kepler_masked, c='k', s=2)
#        plt.title('Kepler data after planet {} masked'.format(planet_letters[planet_to_mask]))
else:
    time_Kepler_masked = time_Kepler
    flux_Kepler_masked = flux_Kepler

# Phase folding

fig = plt.figure()
#fig, axes = plt.subplots(planet_num, 1, figsize=(8, 10), sharex=False)
        
#plt.title('{}'.format(Kepler_name))
	
# setting up the phase fold data
# n.b. these need changing so that they use the fully detrended versions, e.g. flux_TESS{something:end}
#phases_b = np.mod(time - t0is[0]-periods[0]/2, periods[0])/periods[0]
#phases_b_TESS = np.mod(time_TESS - t0is[0]-periods[0]/2, periods[0])/periods[0]
phases_c_Kepler = np.mod(time_Kepler_masked - t0is[planet_num]-periods[planet_num]/2, periods[planet_num])/periods[planet_num]
#arg_b = np.argsort(phases_b)
#gp_mod = soln["gp_pred"] + soln["mean"]
	
# phase fold for planet

ax = plt.gca()
plt.title('{} folded by planet {} period'.format(Kepler_name,planet_letters[planet_num]))
#ax.scatter(phases_b_TESS, flux_TESS, c='k', s=1, label="TESS Data")
ax.scatter(phases_c_Kepler, flux_Kepler_masked, c='darkgrey', s=1, label="Kepler De-trended Data")
#mod_b = soln["light_curves_b"]
#        ax.plot(phases_b[mask][arg_b], mod_b[arg_b]+0.005, color='orange', label="Planet b Model")
#ax.plot(phases_b[mask][arg_b], mod_b[arg_b], color='orange', label="Planet b Model")
ax.legend(fontsize=12)
ax.set_ylabel("De-trended Flux [ppt]")
ax.set_xlabel("Phase")
ax.set_xlim(0, 1)
txt = "Planet {} Period = {:.3f}".format(planet_letters[planet_num],periods[planet_num])
ax.annotate(
        txt,
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
)
##################### PART 3: Build Model with batman #################

# Could frist split them up into parts and use 'find min'-like function as a first guess
#troughs, trough_info = find_peaks(-flux_Kepler_masked, prominence = -0.001, width = 20)

params = batman.TransitParams()
params.t0 = t0is[planet_num]                      #time of inferior conjunction
params.per = periods[planet_num]                    #orbital period
params.rp = radii[planet_num]/R_star[0]             #planet radius (in units of stellar radii)
params.a = a_array[planet_num]/(0.00465047*R_star[0]) #semi-major axis (in units of stellar radii)
params.inc = incls[planet_num]                      #orbital inclination (in degrees)
params.ecc = eccs[planet_num]                       #eccentricity
params.w = 0.0                           #longitude of periastron (in degrees)
params.limb_dark = "nonlinear"             #limb darkening model
params.u = [0.5, 0.1, 0.1, -0.1]           #limb darkening coefficients [u1, u2, u3, u4]

t = time_Kepler_masked                     #times at which to calculate light curve
m = batman.TransitModel(params, t)
flux = m.light_curve(params) 

plt.figure()
plt.scatter(time_Kepler_masked, flux_Kepler_masked, c='k', s=1, label="Kepler De-trended Data")
plt.plot(time_Kepler_masked,flux)
plt.title('{} with overplotted initial batman model'.format(Kepler_name))
#plt.plot(time_Kepler_masked[troughs], flux_Kepler_masked[troughs], "x", c = 'r')


## Calculate theoretical transit duration... Or simply read off model/data

############### PART 4: Minimize chi-squared in vicinity of each transit #########
#
## Define segments... T0 + n*period (+/- enough to catch)
#
## 1st run - wide:
#
#Work out number of transits
current_t0 = params.t0
initial_t0is = []

# Added to make sure it goes right back to start of data
while current_t0 - periods[planet_num] > time_Kepler[0]:
    current_t0 = current_t0 - periods[planet_num]

while current_t0 < time_Kepler[-1]:
    initial_t0is = initial_t0is + [current_t0]
    current_t0 = current_t0 + periods[planet_num]

n_list = range(len(initial_t0is))
t0is[planet_num] = initial_t0is[0]
#
final_t0is = [1]*len(n_list)
final_t0i_cs = [1]*len(n_list)
#
num_transits_done = 0

if large_TTV == True:
    search_width = 1
    step_size = 0.05
if small_TTV == True:
    search_width = 0.2
    step_size=0.01
else:
    search_width = 0.5
    step_size = 0.01
#
##for n in n_list:
##    int_start = t0is[planet_num] + n*periods[planet_num] - 2
##    int_end = t0is[planet_num] + n*periods[planet_num] + 2
##    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
##    
##    int_time = time_Kepler_masked[idx]
##    int_flux = flux_Kepler_masked[idx]
##    
##    t0i_list = np.arange(t0is[planet_num]+n*periods[planet_num]-search_width, t0is[planet_num]+n*periods[planet_num]+search_width, step_size)
##    
##    chi_sq_list = []
##    test_t0is = []    
##    
##    for i in range(len(t0i_list)):
##        params.t0 = t0i_list[i]
##    
##        m = batman.TransitModel(params, int_time)
##        calc_flux = m.light_curve(params) 
##        
##        chi_sq = scipy.stats.chisquare(int_flux, f_exp=calc_flux)
###        chi_sq_list.append((chi_sq[0],params.t0))
##        
###        plt.figure()
###        plt.scatter(int_time, int_flux, c='k', s=2)
###        plt.plot(int_time, calc_flux)
###        txt = "Chi-sq = {}; p-value = {}".format(chi_sq[0], chi_sq[1])
###        plt.annotate(
###                txt,
###                (0, 0),
###                xycoords="axes fraction",
###                xytext=(5, 5),
###                textcoords="offset points",
###                ha="left",
###                va="bottom",
###                fontsize=12,
###        )
##        if chi_sq[0] < final_t0i_cs[n]:
##            final_t0is[n] = params.t0
##            final_t0i_cs[n] = chi_sq[0]
##        print('Number of transits analysed = {}'.format(num_transits_done))
##        num_transits_done += 1
####
final_t0is = np.array(final_t0is)
initial_t0is = np.array(initial_t0is)
####
##with open(Kepler_name + ' {} wide_t0is.pkl'.format(planet_letter), 'wb') as f:
##    pickle.dump(final_t0is,f)
###with open(Kepler_name + ' {} wide_t0is.pkl'.format(planet_letter), 'rb') as f:
###    final_t0is = pickle.load(f)
###
####o_c = final_t0is - initial_t0is
###
###Plot new fold
##folded_fig = plt.figure()
##for n in n_list:
##    int_start = final_t0is[n] - 2
##    int_end = final_t0is[n] + 2
##    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
##    
##    int_time = time_Kepler_masked[idx] - final_t0is[n]
##    int_flux = flux_Kepler_masked[idx] 
##    plt.scatter(int_time, int_flux, s=1, c='k')
##plt.title('Folded fig for {} {} - Iteration 1'.format(Kepler_name, planet_letter))
##plt.xlabel('Time since transit [Days]')
##plt.ylabel('Normalized Flux')
##plt.savefig('{} {} - Iteration 1.png'.format(Kepler_name, planet_letter))
##
##
### 2nd run: 1min step-size
##final_t0is2, o_c2 = find_ttvs(final_t0is, periods[planet_num], time_Kepler_masked, flux_Kepler_masked, params, search_width=0.075, step_size=1/(24*60))  #1min step
##
##with open(Kepler_name + ' {} 1min_t0is.pkl'.format(planet_letter), 'wb') as f:
##    pickle.dump(final_t0is2,f)
###with open(Kepler_name + ' {} 1min_t0is.pkl'.format(planet_letter), 'rb') as f:
###    final_t0is2 = pickle.load(f)
##
##o_c_final = final_t0is2 - initial_t0is
###
###Plot new fold
##folded_fig2 = plt.figure()
##linear_flux = np.array([])
##linear_time = np.array([])
##for n in n_list:
##    int_start = final_t0is2[n] - 2
##    int_end = final_t0is2[n] + 2
##    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
##    
##    int_time = time_Kepler_masked[idx] - final_t0is2[n]
##    int_flux = flux_Kepler_masked[idx] 
##    plt.scatter(int_time, int_flux, s=1, c='k')
##    linear_flux = np.append(linear_flux,int_flux)
##    linear_time = np.append(linear_time,int_time+final_t0is2[n]-o_c_final[n])
##plt.title('Folded fig for {} {} - Iteration 2'.format(Kepler_name, planet_letter))
##plt.xlabel('Time since transit [Days]')
##plt.ylabel('Normalized Flux')
###
##linear_fig = plt.figure()
##phase = np.mod(linear_time-t0is[planet_num]-periods[planet_num]/2,periods[planet_num])/periods[planet_num]
##plt.scatter(phase, linear_flux, c= 'k', s=2)
##plt.title('{} {} data folded after linearisation'.format(Kepler_name, planet_letter))
##plt.xlabel('Phase')
##plt.ylabel('Normalized Flux')
##
if no_TTV == True:
    linear_time = time_Kepler_masked
    linear_flux = flux_Kepler_masked
    final_t0is2 = initial_t0is
#
############################ Model stacked transit #############################
#  
def build_model(mask=None, start=None, optimisation=True):
    if mask is None:
        mask = np.ones(len(time_TESS), dtype=bool)
    with pm.Model() as model:

		############################################################        
		
        ### Stellar parameters

        mean = pm.Normal("mean", mu=1.0, sd=1.0)				#mean = the baseline flux = 0 for the TESS data 
		# you can define a new mean for each set of photometry, just to keep track of it all
		
        u_star = xo.distributions.QuadLimbDark("u_star")				#kipping13 quad limb darkening
		
#        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)			#using a bounded normal places constraints the prob distribution by introducing limits
#        m_star = BoundedNormal("m_star", mu=M_star[0], sd=M_star[1],testval=np.around(M_star[0], decimals = 1))	#stellar mass
#        r_star = BoundedNormal("r_star", mu=R_star[0], sd=R_star[1],testval=np.around(R_star[0], decimals = 1))	#stellar radius
        m_star = M_star[0]
        r_star = R_star[0]*0.98 #*1.3 for KOI-12b
        
		############################################################    
		
        ### Orbital parameters for the planets
        # Planet b
        P_b = pm.Normal("P_b", mu=periods[planet_num], sd=period_sds[planet_num]) #the period (unlogged)
        t0_b = pm.Normal("t0_b", mu=t0is[planet_num], sd=t0i_sds[planet_num])	#time of a ref transit for each planet
        logr_b = pm.Normal("logr_b", mu=np.log(radii[planet_num]), sd=1.0)#log radius - we keep this one as a log
#        logr_b = np.log(1 * radii[planet_num])
        r_pl_b = pm.Deterministic("r_pl_b", tt.exp(logr_b)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
        ratio_b = pm.Deterministic("ror_b", r_pl_b / r_star) #ratio - radius ratio between planet and star    		
        if incls[planet_num] == 90:
            mu_try = 87.5 #incls[planet_num]
            incl_b = pm.Normal('incl',mu=mu_try/180*np.pi, sd=1.0)
        else:
#            incl_b = incls[planet_num]/180*np.pi
            incl_b = pm.Normal('incl',mu=incls[planet_num]/180*np.pi, sd=1.0)
#            incl_b = pm.Normal('incl',mu=86/180*np.pi,sd=1.0)
#        b_mu = a_array[planet_num]*np.cos(incls[planet_num])/R_star[0]
#        b_mu = 0.0774
#        b_b = xo.distributions.ImpactParameter("b_b", ror=ratio_b) # Calculate impact parameter) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
#        b_b = pm.Normal("b_b", mu=abs(b_mu),sd = 0.01)
        ecc_b = eccs[planet_num]
#        ecc_b = pm.Normal("ecc_b", mu=eccs[planet_num], sd=ecc_sds[planet_num])
        omega_b = 0.0            
                
        ############################################################    
	
		### Transit jitter & GP parameters for TESS LIGHTCURVE
	
#        logs2 = pm.Normal("logs2", mu=np.log(np.var(flux_Kepler_masked)), sd=0.05)
#        logw0 = pm.Normal("logw0", mu=0.0, sd=0.05)
#        logSw4 = pm.Normal("logSw4", mu=np.log(np.var(flux_Kepler_masked)), sd=0.05)
		# this sets up a GP for the TESS lightcurve, as per the tutorials.
		# reducing sd seems to make the GP a little less wiggly
        
        ############################################################    

		### Orbit model (Keplerian)
        if method == 'manual':
		#planet b
            orbit_b = xo.orbits.KeplerianOrbit(
                    r_star=r_star, 
                    m_star=m_star,
                    period=P_b,
                    t0=t0_b,
                    incl=incl_b,
                    ecc=ecc_b,
                    omega=omega_b,
                    )
    		#planet c
#            orbit_c = xo.orbits.KeplerianOrbit(
#                    r_star=r_star,
#                    m_star=m_star,
#                    period=P_c,
#                    t0=t0_c,
#                    b=b_c,
#                    ecc=ecc_c,
#                    omega=omega_c,
#                    )
#        elif method == 'array':
#            orbit = xo.orbits.KeplerianOrbit(                    
#                    r_star=r_star,
#                    m_star=m_star,
#                    period=P,
#                    t0=t0,
#                    b=b,
#                    ecc=ecc,
#                    omega=omega)
        else:
            # Planet b
            orbit_b = xo.orbits.KeplerianOrbit(
                    r_star=r_star, 
                    m_star=m_star,
                    period=P_b,
                    t0=t0_b,
                    incl=incl_b,
                    ecc=ecc_b,
                    omega=omega_b,
                    )
	
		############################################################            
        ### Compute the model light curve using starry FOR TESS LIGHTCURVE
		# it seems to break without the *1. 
        

        #planet b		  # n.b. can also use u_star in first bracket
        light_curves_b = (
                xo.LimbDarkLightCurve(params.u).get_light_curve(
                orbit=orbit_b, r=r_pl_b, t=linear_time, texp=texp_Kepler
                )
                * 1
        )
        light_curve_b = pm.math.sum(light_curves_b, axis=-1) + mean 	#this is the eclipse_model
        pm.Deterministic("light_curves_b", light_curves_b) 			#tracking val of model light curve for plots
        
        if planet_num == 1:
            light_curve_TESS = pm.math.sum(light_curves_b, axis=-1) + mean
		
        pm.Normal("obs", mu=light_curve_b, sd=0.1, observed=linear_flux-1)
        
		############################################################    
		
        ### GP model for the light curve
		# Essentially from the tutorial
	
#        kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2)) # n.b. SHOTerm = Stochastically driven, damped Harmonic Osciallator. Other recommended options are Matern32Term (Matern-3/2 function) and RotationTerm (two SHO for stellar rotation)
#        gp = xo.gp.GP(kernel, time_Kepler_masked, tt.exp(logs2) + tt.zeros(mask.sum()))
#        print(flux_TESS[mask])
#        print(light_curve_TESS)
#        pm.Potential("transit_obs", gp.log_likelihood(flux_Kepler_masked - light_curve_TESS))
#        pm.Deterministic("gp_pred", gp.predict())
		
	
		### FITTING SEQUENCE		
		
        if start is None:
            start = model.test_point
        map_soln = xo.optimize(start=start)
        map_soln = xo.optimize(start=map_soln)
        map_soln = xo.optimize(start=map_soln)
		
        trace = []

    return trace, model, map_soln, #vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H  # with RVs, you need to return some extra stuff for plotting

trace, model0, map_soln0 = build_model() # this allows you to reuse the model or something later on - for GPs, add in: vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H

print(map_soln0)          


def plot_light_curve(soln, mask=None):
    if mask is None:
        mask = np.ones(len(linear_time), dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
	
    # this plot shows the og lightcurves with the gp model on top
#    ax = axes[0]
#    ax.scatter(linear_time[mask], linear_flux[mask], c='k', s = 1, label="Original Data")
#    gp_mod = soln["gp_pred"] + soln["mean"]
#    ax.plot(time_Kepler_masked[mask], gp_mod, color="C2", label="GP Model")
#    ax.legend(fontsize=12)
#    ax.set_ylabel("Relative Flux [ppt]")

    # this plot shows the clean lightcurve (og lightcurve - gp solution) plus the light curve models for planets b and c
    ax = axes[0]
    ax.scatter(linear_time[mask], linear_flux[mask]-1, c='k', s=1, label="De-trended Data")
    mod_b = soln["light_curves_b"]
    ax.plot(linear_time[mask], mod_b, color='orange', label="Planet Model")
    ax.legend(fontsize=12)
    ax.set_ylabel("De-trended Flux [ppt]")
	
    # this plot shows the residuals
    ax = axes[1]
    mod = np.sum(soln["light_curves_b"], axis=-1)
    ax.scatter(linear_time[mask], linear_flux[mask] - mod-1, c='k', s=1, label='Residuals')
    ax.axhline(0, color="mediumvioletred", lw=1, label = 'Baseline Flux')
    ax.set_ylabel("Residuals [ppt]")
    ax.legend(fontsize=12)
    ax.set_xlim(linear_time[mask].min(), linear_time[mask].max())
    ax.set_xlabel("Time [BJD - 2454833]")
    plt.title('{}'.format(target_ID))
	
    plt.subplots_adjust(hspace=0)

    return fig

plot_light_curve(map_soln0)

def plot_phase_curve_auto(soln, mask=None,instrument = 'both', planet_num=1, pl_letter=planet_letter):
    if mask is None:
    		mask = np.ones(len(time_TESS), dtype=bool)
    
    fig = plt.figure()
    #plt.title('{}'.format(target_ID))

    # setting up the phase fold data
    phases_b = np.mod(linear_time - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
    arg_b = np.argsort(phases_b)
	
    # phase fold for planet b
    ax = plt.gca()
    ax.scatter(phases_b, linear_flux-1, c='k', s=1, label="De-trended Data")
    binned_time_b, binned_flux_b = binned(phases_b[arg_b], linear_flux[arg_b]-1)
    ax.scatter(binned_time_b, binned_flux_b, c='r', s=1, label="Binned De-trended Data")
    mod_b = soln["light_curves_b"]
    ax.plot(phases_b[arg_b], mod_b[arg_b], color='orange', label="Planet {} Model".format(pl_letter))
    ax.legend(fontsize=12)
    ax.set_ylabel("De-trended Flux [ppt]")
    ax.set_xlabel("Phase")
    ax.set_xlim(0, 1)
    txt = "Planet {} Period = {:.3f}".format(pl_letter, map_soln0['P_b'])
    ax.annotate(
            txt,
            (0, 0),
            xycoords="axes fraction",
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=12,
    )
    fig.savefig('Batman model for {} {}'.format(Kepler_name,pl_letter))
	
    return fig

plot_phase_curve_auto(map_soln0, instrument = instrument)
###
###
#### TTV run 3: 10s step-size
##Update model parameters based on stacked fit
params.rp = map_soln0['r_pl_b']
#params.limb_dark = "quadratic"
#params.u = [map_soln0['u_star'][0],map_soln0['u_star'][1]] 
#
if no_TTV == True:
    final_t0is3 = initial_t0is
else:
    final_t0is3, o_c3 = find_ttvs(final_t0is2, periods[planet_num], time_Kepler_masked, flux_Kepler_masked, params, search_width=0.025, step_size=1/(24*60*6), run_number=3)  #10s step

#Plot new fold
folded_fig3 = plt.figure()
linear_flux = np.array([])
linear_time = np.array([])
for n in n_list:
    int_start = final_t0is3[n] - 2
    int_end = final_t0is3[n] + 2
    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
    
    int_time = time_Kepler_masked[idx] - final_t0is3[n]
    int_flux = flux_Kepler_masked[idx] 
    plt.scatter(int_time, int_flux, s=1, c='k')
#    linear_flux = np.append(linear_flux,int_flux)
#    linear_time = np.append(linear_time,int_time+final_t0is2[n]-o_c_final[n])
plt.title('Folded fig for {} {} - 3rd run'.format(Kepler_name, planet_letter))
plt.xlabel('Time since transit [Days]')
plt.ylabel('Normalized Flux')

########## OR: Set all but T0 and use emcee/Exoplanet to model ####################
#def build_model_piecewise(mask=None, start=None, optimisation=True, test_t0 = 0.0, segment_time=linear_time, segment_flux=linear_flux,f_err=flux_err,texp=texp_Kepler,tune_step=3000,draw_step=3000):
#    if mask is None:
#        mask = np.ones(len(segment_time), dtype=bool)
#    with pm.Model() as model:
#
#		############################################################        
#		
#        ### Stellar parameters
#
#        mean = pm.Normal("mean", mu=1.0, sd=1.0)
#		
#        u_star = [map_soln0['u_star'][0],map_soln0['u_star'][1]]			#kipping13 quad limb darkening
#
##        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)			#using a bounded normal places constraints the prob distribution by introducing limits
#        m_star = M_star[0] #map_soln0['m_star']	#stellar mass
#        r_star = R_star[0] #map_soln0['r_star']	#stellar radius
#
#		############################################################    
#		
#        ### Orbital parameters for the planets
#        # Planet b
#        P_b = periods[planet_num] #map_soln0['P_b'] #the period 
#        t0_b = pm.Normal("t0_b", mu=test_t0, sd=1.0)	#time of a ref transit for each planet
##        t0_b = pm.Uniform("t0_b", upper=test_t0+1, lower=test_t0-1)
#        logr_b = map_soln0["logr_b"]
##        logr_b = np.log(1 * radii[planet_num]) #log radius - we keep this one as a log
#        r_pl_b = map_soln0['r_pl_b']
##        r_pl_b = pm.Deterministic("r_pl_b", tt.exp(logr_b)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
##        ratio_b = pm.Deterministic("ror_b", r_pl_b / r_star) #ratio - radius ratio between planet and star    		
##        b_b = map_soln0['b_b'] # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
#        incl_b = incls[planet_num]/180*np.pi #map_soln0['incl'] #
#        ecc_b = eccs[planet_num]
#        omega_b = 0.0            
#        
#        ############################################################    
#
#        orbit_b = xo.orbits.KeplerianOrbit(
#                r_star=r_star, 
#                m_star=m_star,
#                period=P_b,
#                t0=t0_b,
#                incl = incl_b,
#                ecc=ecc_b,
#                omega=omega_b,
#                )
#	
#		############################################################            
#        ### Compute the model light curve using starry FOR TESS LIGHTCURVE        
#
#        #planet b		
#        light_curves_b = (
#                xo.LimbDarkLightCurve(params.u).get_light_curve(
#                orbit=orbit_b, r=r_pl_b, t=segment_time, texp=texp
#                )
#                * 1
#        )
#        light_curve_b = pm.math.sum(light_curves_b, axis=-1) + mean 	#this is the eclipse_model
#        pm.Deterministic("light_curves_b", light_curves_b) 			#tracking val of model light curve for plots
#		
#        pm.Normal("obs", mu=light_curve_b, sd=f_err, observed=segment_flux)
#		
#	
#		### FITTING SEQUENCE		
#        if optimisation == True:
#            if start is None:
#                start = model.test_point
#            map_soln = xo.optimize(start=start)
#            map_soln = xo.optimize(start=map_soln)
#            map_soln = xo.optimize(start=map_soln)
#        else:
#            map_soln=model.test_point
#		
###        trace = []
##        model_t0 = map_soln['t0_b']
#        
#        # n.b. in Edwards+ they used 30,000 burn-in (tuning) and 100,000 iterations (draws), with 200 walkers
#        trace = pm.sample(
#        tune=tune_step, #Previously both were 5000
#        draws=draw_step,
#        start=map_soln,
#        cores=2,
#        chains=2,
#        step=xo.get_dense_nuts_step(target_accept=0.9), #Previously 0.9
#        )
#        
##        summary_df =pm.summary(trace, varnames=["t0_b","mean"]) #n.b. check pymc3 for better thing than summary - it's probably concatenating it
##        sd = summary_df.sd['t0_b']
#        sd = np.std(trace["t0_b"])
#        model_t0 = np.mean(trace["t0_b"])
##    
##        pm.summary(trace, varnames=["t0_b","mean"])
###        pm.plot_trace(trace, varnames=["t0_b","mean"])
##        samples = pm.trace_to_dataframe(trace, varnames=["t0_b", "mean"])
##        truth = np.concatenate(
##            xo.eval_in_model([t0_b, mean], model.test_point, model=model)
##        )
##        _ = corner.corner(
##            samples,
##            truths=truth,
##            labels=["t0_b", "mean"]
##        )
#
#    return model, map_soln, model_t0, sd, trace # vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H  # with RVs, you need to return some extra stuff for plotting
##
#model_t0s = np.array([])
#sds = np.array([])
##
##n_list = [0]
##n_list = [713,714]
#
#for n in n_list: #n.b. got up to 713
#    int_start = final_t0is3[n] -1
#    int_end = final_t0is3[n] + 1
#    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
#    
#    int_time = time_Kepler_masked[idx]
#    int_flux = flux_Kepler_masked[idx]
#    t0 = final_t0is3[n]
#    
#    #    plt.figure()
#    #    plt.scatter(int_time, int_flux-1, c='k', s=1, label="De-trended Data")
#    #    plt.axvline(x=t0, c='r')
#    #    plt.legend(fontsize=12)
#    #    plt.ylabel("De-trended Flux [ppt]")
#    
#    model1, map_soln1, model_t0, sd, trace = build_model_piecewise(test_t0=t0, segment_time=int_time, segment_flux=int_flux,tune_step=1000,draw_step=1000) # this allows you to reuse the model or something later on - for GPs, add in: vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H
#
#    model_t0s = np.append(model_t0s,model_t0)
#    sds = np.append(sds,sd)
#    print('Number of transits fit = {}'.format(n+1))
##print(map_soln1)
##
#plt.figure()
#plt.scatter(int_time, int_flux-1, c='k', s=1, label="De-trended Data")
#mod_b = map_soln1["light_curves_b"]
#plt.plot(int_time, mod_b, color='orange', label="Planet b Model")
#plt.legend(fontsize=12)
#plt.ylabel("De-trended Flux [ppt]")
#
#
##np.random.seed(42)
##with model1:
##    trace = pm.sample(
##        tune=5000,
##        draws=5000,
##        start=map_soln1,
##        cores=2,
##        chains=2,
##    )
##    
#pm.summary(trace, varnames=["t0_b","mean"])
#pm.plot_trace(trace, varnames=["t0_b","mean"])
##
##thing=pm.summary(trace, varnames=["t0_b","mean"])
##thing.sd['t0_b']
#
##samples = pm.trace_to_dataframe(trace, varnames=["t0_b", "mean"])
##truth = np.concatenate(
##    xo.eval_in_model([t0_b, mean], model0.test_point, model=model0)
##)
##_ = corner.corner(
##    samples,
##    truths=truth,
##    labels=["t0_b", "mean"]
##)
#
#
##with open(save_path + 'Final t0s/Kepler t0s for {} {}.csv'.format(Kepler_name, planet_letter), 'r') as read_obj:
##    csv_reader = csv.reader(read_obj)
##    model_t0s = next(csv_reader)
##    sds = next(csv_reader)
##
##model_t0s = np.array([float(data) for data in model_t0s])
##sds = np.array([float(data) for data in sds])
#
## Mask dodgy ones (usually those which fall outside data windows)
#err = np.array([0]*len(n_list))
#
#mask = np.ones(len(n_list), dtype=bool)
#for n in n_list:
##    err[n] = np.sqrt(sds[n]**2 + period_sds[planet_num]**2 + n**2*t0i_sds_real[planet_num]**2)
#    err[n] = sds[n]
#    if sds[n] > 0.008:
#        mask[n] = False
#
#with open(save_path + 'Final t0s/Kepler t0s for {} {}.csv'.format(Kepler_name, planet_letter), 'w') as csv_file:
#    csv_writer = csv.writer(csv_file, delimiter=',')
#    csv_writer.writerow(model_t0s)
#    csv_writer.writerow(sds)
#
######################## Find new period and T0: ##############################
#transit_ns = np.array([round((item-t0is[planet_num])/periods[planet_num]) for item in model_t0s])
#
##Unweihgted fit
#Kepler_period, Kepler_fit_t0 = np.polyfit(transit_ns[mask], model_t0s[mask],1)
#y_model = np.array(transit_ns)*Kepler_period + Kepler_fit_t0
#
## Weighted fit
##def f(n, t0, per):
##    """ Straight line ephemeris fit for planet with period=per and t0  """
##    return t0 + per*n
##popt, pcov = curve_fit(f, n, per, sigma=sds[mask], absolute_sigma=True)
##y_model = f(x, *popt)
##Kepler_fit_period = popt[0]
##Kepler_fit_t0 = popt[1]
#
#plt.figure()
#plt.errorbar(transit_ns[mask], model_t0s[mask], yerr=sds[mask], fmt='.', c='k')
#plt.plot(transit_ns,y_model)
#
#current_t0 = Kepler_fit_t0
#new_t0is = []
#
#while current_t0 < time_Kepler[-1]:
#    new_t0is.append(current_t0)
#    current_t0 = current_t0 + Kepler_period
#
##Also plot unmasked version...
#
################## PART 5: Compute and plot O-C diagram #####################
##Using final observed values compute O-C values for all transits and plot
##o_c_final = final_t0is3 - initial_t0is
##o_c_final = model_t0s - initial_t0is
#o_c_final = model_t0s - new_t0is
#o_c_hrs = o_c_final*24
#
#o_c_masked = o_c_hrs[mask]
#model_t0s_masked = model_t0s[mask]
#
##e=[24*0.004]*len(final_t0is3)
##e = 24*err
#e = 24*sds
#
#un_masked_o_c_plot = plt.figure()
##plt.scatter(final_t0is3, o_c_hrs, c='k',s=1)
#plt.errorbar(model_t0s, o_c_hrs, yerr=e, fmt='.', c='k')
#plt.xlabel('BJD Time [Days]')
#plt.ylabel('O-C [hrs]')
#plt.title('Unmaksed O-C diagram for {} {} after individual exoplanet fit'.format(Kepler_name, planet_letter))
#
#o_c_plot = plt.figure()
##plt.scatter(final_t0is3, o_c_hrs, c='k',s=1)
#plt.errorbar(model_t0s_masked, o_c_masked, yerr=e[mask], fmt='.', c='k')
#plt.xlabel('BJD Time [Days]')
#plt.ylabel('O-C [hrs]')
#plt.title('O-C diagram for {} {} after individual exoplanet fit'.format(Kepler_name, planet_letter))
#
#
## Initial compared to final positions
#plt.figure()
#plt.scatter(time_Kepler_masked, flux_Kepler_masked, c='k', s=1, label="Kepler De-trended Data")
#for t0i in initial_t0is:
#    plt.axvline(x=t0i, c='b')
#for t0i in model_t0s:
#    plt.axvline(x=t0i, c='r')
#plt.axvline(x=initial_t0is[0], c='b', label='Initial T0')
#plt.axvline(x=model_t0s[0], c='r', label='Final T0')
#plt.legend()
#plt.title('{} {} initial and final T0s after one run'.format(Kepler_name, planet_letter))
#
#
#folded_fig4 = plt.figure()
#linear_flux = np.array([])
#linear_time = np.array([])
#for n in n_list:
#    int_start = model_t0s[n] - 2
#    int_end = model_t0s[n] + 2
#    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
#    
#    int_time = time_Kepler_masked[idx] - final_t0is3[n]
#    int_flux = flux_Kepler_masked[idx] 
#    plt.scatter(int_time, int_flux, s=1, c='k')
##    linear_flux = np.append(linear_flux,int_flux)
##    linear_time = np.append(linear_time,int_time+final_t0is2[n]-o_c_final[n])
#plt.title('Folded fig for {} {} - Model run'.format(Kepler_name, planet_letter))
#plt.xlabel('Time since transit [Days]')
#plt.ylabel('Normalized Flux')
#
################################ PART 6: TESS Stuff ##############################
###
plt.figure()
plt.scatter(time_TESS,flux_TESS,c='k',s=1)

# Calculate expected TESS times
transit_time = t0is[planet_num]
calc_per = periods[planet_num] 
TESS_t0s = np.array([])
model_TESS_t0s = np.array([])
sds_TESS = np.array([])
n = 0

while transit_time < time_TESS[0]:
    transit_time += calc_per
    n += 1

while transit_time < time_TESS[-1]:
    TESS_t0s = np.append(TESS_t0s,transit_time)
    transit_time += calc_per

if transit_mask_TESS == True:
    for n in range(len(TESS_t0s)):
        int_start = TESS_t0s[n] - 2
        int_end = TESS_t0s[n] + 2
        idx = np.where((time_TESS > int_start) & (time_TESS < int_end))

        int_time = time_TESS[idx]
        int_flux = flux_TESS[idx]
        if len(int_time) != 0:
            epoch = TESS_t0s[n]
            period = periods[planet_num]
            duration = 0.5 #Or: use calculated duration x 2-4
            phase = np.mod(int_time-epoch-period/2,period)/period
            
            near_transit = [False]*len(int_flux)
            for i in range(len(int_time)):
                if abs(phase[i] - 0.5) < duration/period:
                    near_transit[i] = True
            near_transit = np.array(near_transit)
            
            t_masked = int_time[~near_transit]
            flux_masked = int_flux[~near_transit]
        #    flux_err_masked = flux_err_cut[~near_transit]
            t_new = int_time[near_transit]
            
            gradient, intercept = np.polyfit(t_masked,flux_masked,1)
            flux_new = t_new*gradient + intercept
            
            interpolated_fig = plt.figure()
            plt.scatter(t_masked, flux_masked, s = 2, c = 'k')
        #    plt.scatter(time_Kepler, flux_Kepler, s = 8, c = 'k')
            plt.scatter(t_new,flux_new, s=2, c = 'r')
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel('Relative flux')
        #    interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
            
            t_transit_mask = np.concatenate((t_masked,t_new), axis = None)
            flux_transit_mask = np.concatenate((flux_masked,flux_new), axis = None)
            
            sorted_order = np.argsort(t_transit_mask)
            t_transit_mask = t_transit_mask[sorted_order]
            flux_transit_mask = flux_transit_mask[sorted_order]
            
            full_transit_mask_time = time_TESS.copy()
            full_transit_mask_flux = flux_TESS.copy()
            full_transit_mask_time[idx] = t_transit_mask
            full_transit_mask_flux[idx] = flux_transit_mask
        else:
            print('Skipped gap at {}'.format(TESS_t0s[n]))
    plt.figure()
    plt.scatter(time_TESS,flux_TESS,c='k',s=1)
    plt.title('TESS lc after transit mask')

# Detrend TESS flux using lowess detrending
if transit_mask_TESS == True:
    detrended_masked_flux, full_lowess_flux = lowess_detrending(flux=full_transit_mask_flux, time=full_transit_mask_time, target_ID=Kepler_name,n_bins=96)
    flux_TESS = flux_TESS/full_lowess_flux
    plt.figure()
    plt.scatter(time_TESS,flux_TESS,c='k',s=1)
    plt.title('TESS lc after transit mask detrend')
else:
    detrended_TESS_flux, full_lowess_flux = lowess_detrending(flux=flux_TESS, time=time_TESS,target_ID=Kepler_name,n_bins=96)
    flux_TESS = detrended_TESS_flux

##TESS_t0s = np.array([2458736.5])
## Fit each TESS Transit
#for n in range(len(TESS_t0s)):
#    int_start = TESS_t0s[n] - 2
#    int_end = TESS_t0s[n] + 2
#    idx = np.where((time_TESS > int_start) & (time_TESS < int_end))
#    
#    int_time = time_TESS[idx]
#    int_flux = flux_TESS[idx]
#    t0 = TESS_t0s[n]
#    if len(int_time) != 0:
#        model_TESS, map_soln_TESS, model_t0, sd, trace = build_model_piecewise(test_t0=t0, segment_time=int_time, segment_flux=int_flux, f_err=mean_flux_err_TESS, texp=texp_TESS,tune_step=10000,draw_step=10000) # this allows you to reuse the model or something later on - for GPs, add in: vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H
#        model_TESS_t0s = np.append(model_TESS_t0s,model_t0)
#        
#        plt.figure()
#        plt.scatter(int_time, int_flux-1, c='k', s=1, label="De-trended Data")
#        binned_TESS_int_time, binned_TESS_int_flux = binned(int_time, int_flux, binsize=15)
#        plt.scatter(binned_TESS_int_time, binned_TESS_int_flux-1, c='r',s=1,label="Binned TESS Data")
#        plt.axvline(x=t0, c='r')
#        plt.legend(fontsize=12)
#        plt.ylabel("De-trended Flux [ppt]")
#        
#        sds_TESS = np.append(sds_TESS,sd)
#        print('Number of transits fit = {}'.format(n+1))
#    else:
#        print('Skipped gap at {}'.format(TESS_t0s[n]))
#
#
#with open(save_path + 'Final t0s/TESS t0s for {} {}.csv'.format(Kepler_name,planet_letter), 'w') as csv_file:
#    csv_writer = csv.writer(csv_file, delimiter=',')
#    csv_writer.writerow(model_TESS_t0s)
#    csv_writer.writerow(sds_TESS)
#
#if len(int_time) != 0:
#    binned_TESS_int_time, binned_TESS_int_flux = binned(int_time, int_flux, binsize=15)
#
#    plt.figure()
#    plt.scatter(int_time, int_flux-1, c='k', s=1, label="De-trended Data")
#    plt.scatter(binned_TESS_int_time, binned_TESS_int_flux-1, c='r',s=1,label="Binned TESS Data")
#    mod_b = map_soln_TESS["light_curves_b"]
#    plt.plot(int_time, mod_b, color='orange', label="Planet {} Model".format(planet_letter))
#    plt.legend(fontsize=12)
#    plt.ylabel("De-trended Flux [ppt]")
#
#pm.plot_trace(trace, varnames=["t0_b","mean"])
#thing=pm.summary(trace, varnames=["t0_b","mean"])
#
#mask_TESS = np.ones(len(model_TESS_t0s), dtype=bool)
##mask_TESS[-1] = False
##
### Find new period and T0: 
#transit_ns_TESS = np.array([round((item-t0is[planet_num])/periods[planet_num]) for item in model_TESS_t0s])
#overall_transit_ns = np.append(transit_ns[mask], transit_ns_TESS[mask_TESS])
#overall_model_t0s = np.append(model_t0s[mask], model_TESS_t0s[mask_TESS])
#overall_err = np.append(sds[mask],sds_TESS[mask_TESS])
#
#with open(save_path + 'Final t0s/Overall t0s for {} {}.csv'.format(Kepler_name, planet_letter), 'w') as csv_file:
#    csv_writer = csv.writer(csv_file, delimiter=',')
#    csv_writer.writerow(overall_model_t0s)
#    csv_writer.writerow(overall_err)
#    
########## n.b. polyfit is pretty simple - change to weighted scipy.optimize.curve_fit instead!
#combined_period, combined_fit_t0 = np.polyfit(overall_transit_ns, overall_model_t0s,1)
#y_model_overall = np.array(overall_transit_ns)*combined_period + combined_fit_t0
#
##popt, pcov = curve_fit(f, overall_transit_ns, overall_model_t0s, sigma=overall_err, absolute_sigma=True)
##y_model = f(overall_model_t0s, *popt)
##Kepler_fit_period = popt[0]
##Kepler_fit_t0 = popt[1]
#
#plt.figure()
#plt.errorbar(overall_transit_ns, overall_model_t0s, yerr=overall_err, fmt='.', c='k')
#plt.plot(overall_transit_ns,y_model_overall)
#plt.xlabel('Transits since T0')
#plt.ylabel('BJD Time [Days]')
#
## Calculate TESS O-C and replot entire O-C diagram
##o_c_TESS = model_TESS_t0s - TESS_t0s
##o_c_hrs_TESS = o_c_TESS*24
##e_TESS = 24*sds_TESS
#    
## Calculate overall O-C and replot entire O-C diagram
#o_c_combined = overall_model_t0s - y_model_overall
#o_c_hrs_combined = o_c_combined*24
#e_overall_hrs = 24*overall_err
#
#o_c_plot_final = plt.figure()
##plt.scatter(final_t0is3, o_c_hrs, c='k',s=1)
##plt.errorbar(model_t0s_masked, o_c_masked, yerr=e[mask], fmt='.', c='k')
##plt.errorbar(model_TESS_t0s, o_c_hrs_TESS, yerr=e_TESS,  fmt='.', c='k')
#plt.errorbar(overall_model_t0s, o_c_hrs_combined, yerr=e_overall_hrs, fmt='.', c='k')
#plt.xlabel('BJD Time [Days]')
#plt.ylabel('O-C [hrs]')
#plt.title('O-C diagram for {} {} including TESS'.format(Kepler_name, planet_letter))
#plt.show()

#### Fit overall ephemerides for systems with flat O-C diagrams

#def model_flat_systems(mask=None, start=None, optimisation=True, test_t0 = 0.0, segment_time=linear_time, segment_flux=linear_flux,f_err=flux_err,texp=texp_Kepler,tune_step=3000,draw_step=3000):
#    if mask is None:
#        mask = np.ones(len(segment_time), dtype=bool)
#    with pm.Model() as model:
#
#		############################################################        
#		
#        ### Stellar parameters
#
#        mean = pm.Normal("mean", mu=1.0, sd=1.0)
#		
##        u_star = [map_soln0['u_star'][0],map_soln0['u_star'][1]]			#kipping13 quad limb darkening
#
##        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)			#using a bounded normal places constraints the prob distribution by introducing limits
#        m_star = M_star[0] #map_soln0['m_star']	#stellar mass
#        r_star = R_star[0] #map_soln0['r_star']	#stellar radius
#
#		############################################################    
#		
#        ### Orbital parameters for the planets
#        # Planet b
#        P_b = pm.Normal("P_b",mu =periods[planet_num], sd=1.0) #map_soln0['P_b'] #the period 
#        t0_b = pm.Normal("t0_b", mu=test_t0, sd=1.0)	#time of a ref transit for each planet
##        t0_b = pm.Uniform("t0_b", upper=test_t0+1, lower=test_t0-1)
#        logr_b = map_soln0["logr_b"]
##        logr_b = np.log(1 * radii[planet_num]) #log radius - we keep this one as a log
#        r_pl_b = map_soln0['r_pl_b']
##        r_pl_b = pm.Deterministic("r_pl_b", tt.exp(logr_b)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
##        ratio_b = pm.Deterministic("ror_b", r_pl_b / r_star) #ratio - radius ratio between planet and star    		
##        b_b = map_soln0['b_b'] # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
#        incl_b = map_soln0['incl'] #incls[planet_num]/180*np.pi # #
#        ecc_b = eccs[planet_num]
#        omega_b = 0.0            
#        
#        ############################################################    
#
#        orbit_b = xo.orbits.KeplerianOrbit(
#                r_star=r_star, 
#                m_star=m_star,
#                period=P_b,
#                t0=t0_b,
#                incl = incl_b,
#                ecc=ecc_b,
#                omega=omega_b,
#                )
#	
#		############################################################            
#        ### Compute the model light curve using starry FOR TESS LIGHTCURVE        
#
#        #planet b		
#        light_curves_b = (
#                xo.LimbDarkLightCurve(params.u).get_light_curve(
#                orbit=orbit_b, r=r_pl_b, t=segment_time, texp=texp
#                )
#                * 1
#        )
#        light_curve_b = pm.math.sum(light_curves_b, axis=-1) + mean 	#this is the eclipse_model
#        pm.Deterministic("light_curves_b", light_curves_b) 			#tracking val of model light curve for plots
#		
#        pm.Normal("obs", mu=light_curve_b, sd=f_err, observed=segment_flux)
#		
#	
#		### FITTING SEQUENCE		
#        if optimisation == True:
#            if start is None:
#                start = model.test_point
#            map_soln = xo.optimize(start=start)
#            map_soln = xo.optimize(start=map_soln)
#            map_soln = xo.optimize(start=map_soln)
#        else:
#            map_soln=model.test_point
#		
###        trace = []
##        model_t0 = map_soln['t0_b']
#        
#        # n.b. in Edwards+ they used 30,000 burn-in (tuning) and 100,000 iterations (draws), with 200 walkers
#        trace = pm.sample(
#        tune=tune_step, #Previously both were 5000
#        draws=draw_step,
#        start=map_soln,
#        cores=2,
#        chains=2,
#        step=xo.get_dense_nuts_step(target_accept=0.9), #Previously 0.9
#        )
#        
##        summary_df =pm.summary(trace, varnames=["t0_b","mean"]) #n.b. check pymc3 for better thing than summary - it's probably concatenating it
##        sd = summary_df.sd['t0_b']
#        model_t0 = np.mean(trace["t0_b"])
#        t0_sd = np.std(trace["t0_b"])
#        model_per = np.mean(trace["P_b"])
#        per_sd = np.std(trace["P_b"])
##    
##        pm.summary(trace, varnames=["t0_b","mean"])
###        pm.plot_trace(trace, varnames=["t0_b","mean"])
##        samples = pm.trace_to_dataframe(trace, varnames=["t0_b", "mean"])
##        truth = np.concatenate(
##            xo.eval_in_model([t0_b, mean], model.test_point, model=model)
##        )
##        _ = corner.corner(
##            samples,
##            truths=truth,
##            labels=["t0_b", "mean"]
##        )
#
#    return model, map_soln, model_t0, t0_sd, model_per, per_sd, trace # vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H  # with RVs, you need to return some extra stuff for plotting
##
#if no_TTV == True:
#    t0 = map_soln0['t0_b']
#    overall_time = np.append(time_Kepler_masked, time_TESS)
#    overall_flux = np.append(flux_Kepler_masked, flux_TESS)
#    model1, map_soln1, model_t0, t0_sd, model_per, per_sd, trace = model_flat_systems(test_t0=t0, segment_time=overall_time, segment_flux=overall_flux,tune_step=1000,draw_step=5000)
#    print('t0 = {} +/- {}'.format(model_t0, t0_sd))
#    print('Per = {} +/- {}'.format(model_per, per_sd))
#
#    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
#    	
#    # this plot shows the og lightcurves with the gp model on top
#    #    ax = axes[0]
#    #    ax.scatter(linear_time[mask], linear_flux[mask], c='k', s = 1, label="Original Data")
#    #    gp_mod = soln["gp_pred"] + soln["mean"]
#    #    ax.plot(time_Kepler_masked[mask], gp_mod, color="C2", label="GP Model")
#    #    ax.legend(fontsize=12)
#    #    ax.set_ylabel("Relative Flux [ppt]")
#    
#    # this plot shows the clean lightcurve (og lightcurve - gp solution) plus the light curve models for planets b and c
#    ax = axes[0]
#    ax.scatter(overall_time, overall_flux-1, c='k', s=1, label="De-trended Data")
#    mod_b = map_soln1["light_curves_b"]
#    ax.plot(overall_time, mod_b, color='orange', label="Planet Model")
#    ax.legend(fontsize=12)
#    ax.set_ylabel("De-trended Flux [ppt]")
#    	
#    # this plot shows the residuals
#    ax = axes[1]
#    mod = np.sum(map_soln1["light_curves_b"], axis=-1)
#    ax.scatter(overall_time, overall_flux - mod-1, c='k', s=1, label='Residuals')
#    ax.axhline(0, color="mediumvioletred", lw=1, label = 'Baseline Flux')
#    ax.set_ylabel("Residuals [ppt]")
#    ax.legend(fontsize=12)
#    ax.set_xlim(overall_time.min(), overall_time.max())
#    ax.set_xlabel("Time [BJD]")
#    plt.title('{}'.format(target_ID))
#
#def append_list_as_row(file_name, list_of_elem):
#    with open(file_name, 'a+', newline='') as write_obj:
#        csv_writer = csv.writer(write_obj)
#        csv_writer.writerow(list_of_elem)
#
##filename = '/Users/mbattley/Documents/PhD/Kepler-2min xmatch/Final_Ephemerides.csv'
#append_list_as_row('/Users/mbattley/Documents/PhD/Kepler-2min xmatch/Final_Ephemerides.csv',[Kepler_name+planet_letter,model_per,per_sd,model_t0,t0_sd])

end = timing.time()
print('Elapsed time = {}s'.format(end - start))       