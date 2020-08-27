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
import batman
import scipy
#import time
import csv
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


def find_ttvs(initial_t0is, period, time, flux, params):
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
        
        t0i_list = np.arange(initial_t0is[n]-0.075,initial_t0is[n]+0.075,1/(24*60))
        
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
            print('Number of transits analysed (round 2) = {}'.format(num_transits_done))
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

############################## PART 0: Setup ##################################
# lc parameters
save_path = '/Users/mbattley/Documents/PhD/Kepler-2min xmatch/'
target_ID = 'TIC 120571842' # Kepler 25 test-case: 'TIC 120960812' #TIC number
TIC = int(target_ID[4:])
sector = 14
multi_sector = False
planet_data = Table.read(save_path + 'Kepler_planets_reobserved_in_TESS_2min.csv', format='ascii.csv')
#target_ID_list = np.array(pc_data['TICID'])
i = list(planet_data['TICID']).index(int(target_ID[3:]))
instrument = 'both'
planet_letter = ['b','c','d','e','f','g','h']
method = 'auto' #Can be 'array, 'auto' or 'manual'
transit_mask = False
detrending = 'lowess_full'

################### PART 1: Downloading/Opening Light-curves ##################
# TESS 2min
#if (planet_data['S14'][i] != 0) and (planet_data['S15'][i] != 0):
#    multi_sector = [14,15]
##    if (planet_data['S26'][i] != 0):
##        multi_sector = [14,15]
#elif (planet_data['S14'][i] != 0) and (planet_data['S26'][i] != 0):
#    multi_sector = [14,26]
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
#
## Kepler
#KIC = planet_data['KIC'][i]
#lcfs = lightkurve.search_lightcurvefile(KIC, mission='Kepler').download_all()
#stitched_lc = lcfs.PDCSAP_FLUX.stitch()
##stitched_lc = lcfs.PDCSAP_FLUX.stitch(corrector_func=my_custom_corrector_func)
#nancut = np.isnan(stitched_lc.flux) | np.isnan(stitched_lc.time)
#stitched_lc = stitched_lc[~nancut]
#time_Kepler = np.array(stitched_lc.time) + 2454833 #Convert to BJD for consistency
#flux_Kepler = np.array(stitched_lc.flux)/np.median(stitched_lc.flux) -1
#binned_Kepler_time, binned_Kepler_flux = binned(time_Kepler,flux_Kepler,binsize=2)
##yerr = stitched_lc.flux_err
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
#
#kepler_9_data = {'time_combined':time_TESS, 'flux_combined':flux_TESS, 'time_TESS':time_TESS_orig, 'flux_TESS':flux_TESS_orig, 'time_Kepler':time_Kepler, 'flux_Kepler':flux_Kepler }
#with open('Kepler_9_data.pkl', 'wb') as f:
#    pickle.dump(kepler_9_data,f)

with open('Kepler_9_data.pkl', 'rb') as f:
    kepler_9_data_reopened = pickle.load(f)
    time = kepler_9_data_reopened['time_combined'] 
    flux = kepler_9_data_reopened['flux_combined'] +1
    time_TESS = kepler_9_data_reopened['time_TESS'] 
    flux_TESS = kepler_9_data_reopened['flux_TESS'] +1
    time_Kepler = kepler_9_data_reopened['time_Kepler']
    flux_Kepler = kepler_9_data_reopened['flux_Kepler'] +1

time = time_Kepler - 2454833
flux = flux_Kepler

plt.figure()  
plt.scatter(time,flux, c='k', s=1)
plt.xlabel("Time [BJD - 2454833]")
plt.ylabel("Normalized Flux [ppt]")

#################### PART 1b: Detrending lightcurve ###########################

if detrending == 'lowess_full':
    #t_cut = lc_30min.time
    #flux_cut = combined_flux
    full_lowess_flux = np.array([])
    if transit_mask == True:
        lowess = sm.nonparametric.lowess(flux_transit_mask, t_transit_mask, frac=0.02)
    else:
        lowess = sm.nonparametric.lowess(flux, time, frac=0.001)
    
    overplotted_lowess_full_fig = plt.figure()
    plt.scatter(time,flux, c = 'k', s = 2)
    plt.plot(lowess[:, 0], lowess[:, 1])
    plt.title('{} lc with overplotted lowess full lc detrending'.format(target_ID))
    plt.xlabel('Time [BJD]')
    plt.ylabel('Relative flux')
    #overplotted_lowess_full_fig.savefig(save_path + "{} lc with overplotted LOWESS full lc detrending.png".format(target_ID))
    plt.show()
#    plt.close(overplotted_lowess_full_fig)
    
    residual_flux_lowess = flux/lowess[:,1]
    full_lowess_flux = np.concatenate((full_lowess_flux,lowess[:,1]))
    
    lowess_full_residuals_fig = plt.figure()
    plt.scatter(time,residual_flux_lowess, c = 'k', s = 2)
    plt.title('{} lc after lowess full lc detrending'.format(target_ID))
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

##################### PART 2: Planet Parameters ############################
texp_TESS = 120                      # Kepler (60s)/TESS (120s) exposure time (s)
texp_TESS /= 60.0 * 60.0 * 24.0 	     # converting exposure time to days (important!!!!!!)

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
else:
    planet_num =0
    incl = 87.0
    ecc = 0
    planet_params = Table(names=('periodi','periodi_sd','t0i','t0i_sd','radi','a','incl','ecc'))
    for j in range(len(planet_data['TICID'])):
        if planet_data['TICID'][j] == TIC and planet_data['pl_discmethod'][j] == 'Transit':
            if ma.is_masked(planet_data['Per'][j]) == False:
                periodi = planet_data['Per'][j]
                period_sd = planet_data['e_Per'][j]
                t0i = planet_data['T0'][j]
            else:
                periodi = planet_data['pl_orbper'][j]
                period_sd = planet_data['pl_orbpererr1'][j]
                t0i = planet_data['pl_tranmid'][j]
            t0i_sd = 0.1
#            if planet_num == 1:
#                t0i_sd = 0.05
            if ma.is_masked(planet_data['Rp'][j]) == False:
                radi = planet_data['Rp'][j]*const.R_earth.value/const.R_sun.value
            else:
                radi = planet_data['pl_radj'][j]*const.R_jup.value/const.R_sun.value
            #n.b. if getting it from revised info convert from Earth radii to jupiter radii
            M_star = planet_data['st_mass'][j],planet_data['st_masserr1'][j]
            # Gets updated R_star based on DR2 (Berger+16) if available, otherwise falls back to exo archive
            if ma.is_masked(planet_data['R*'][j]) == False:
                R_star = planet_data['R*'][j], planet_data['E_R*_2'][j]
            else:
                R_star = planet_data['st_rad'][j], planet_data['st_raderr1'][j]
            if ma.is_masked(planet_data['pl_orbsmax'][j]) == False:
                a = planet_data['pl_orbsmax'][j]
            if ma.is_masked(planet_data['pl_orbincl'][j]) == False:
                incl = planet_data['pl_orbincl'][j]
            if ma.is_masked(planet_data['pl_orbeccen'][j]) == False:
                ecc = planet_data['pl_orbeccen'][j]
            planet_params.add_row((periodi, period_sd, t0i, t0i_sd, radi, a, incl, ecc))
            planet_num += 1
    periods = np.array(planet_params['periodi'])
    period_sds = np.array(planet_params['periodi_sd'])
    t0is = np.array(planet_params['t0i'])
    t0i_sds = np.array(planet_params['t0i_sd'])
    radii = np.array(planet_params['radi'])
    a_array = np.array(planet_params['a'])
    incls = np.array(planet_params['incl'])
    eccs = np.array(planet_params['ecc'])

############## PART 2b: Masking other planets and plotting phase ##############
# Planet masking
transit_mask = True
if transit_mask == True:
    period = periods[1]
    epoch = t0is[1]
    duration = 1.5
    phase = np.mod(time_Kepler-epoch-period/2,period)/period
    
    plt.figure()
    plt.scatter(phase, flux_Kepler, c= 'k', s=2)
    plt.title('{} data folded by planet c period'.format(target_ID))
    
    near_transit = [False]*len(flux_Kepler)
    
    for i in range(len(time_Kepler)):
        if abs(phase[i] - 0.5) < duration/period:
            near_transit[i] = True
    
    near_transit = np.array(near_transit)
    
    time_Kepler_masked = time_Kepler[~near_transit]
    flux_Kepler_masked = flux_Kepler[~near_transit]

plt.figure()
plt.scatter(time_Kepler_masked, flux_Kepler_masked, c='k', s=2)
plt.title('Kepler data after planet c masked')

# Phase folding
planet_num = 1

if planet_num == 1:
    fig = plt.figure()
#        if planet_num == 6:
#            fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=False)
else:
    fig, axes = plt.subplots(planet_num, 1, figsize=(8, 10), sharex=False)
        
plt.title('{}'.format(target_ID))
	
# setting up the phase fold data
# n.b. these need changing so that they use the fully detrended versions, e.g. flux_TESS{something:end}
#phases_b = np.mod(time - t0is[0]-periods[0]/2, periods[0])/periods[0]
#phases_b_TESS = np.mod(time_TESS - t0is[0]-periods[0]/2, periods[0])/periods[0]
phases_b_Kepler = np.mod(time_Kepler_masked - t0is[0]-periods[0]/2, periods[0])/periods[0]
#arg_b = np.argsort(phases_b)
#gp_mod = soln["gp_pred"] + soln["mean"]
	
# phase fold for planet b
if planet_num == 1:
    ax = plt.gca()
else:
    ax = axes[0]
plt.title('{}'.format(target_ID))
#ax.scatter(phases_b_TESS, flux_TESS, c='k', s=1, label="TESS Data")
ax.scatter(phases_b_Kepler, flux_Kepler_masked, c='darkgrey', s=1, label="Kepler De-trended Data")
#mod_b = soln["light_curves_b"]
#        ax.plot(phases_b[mask][arg_b], mod_b[arg_b]+0.005, color='orange', label="Planet b Model")
#ax.plot(phases_b[mask][arg_b], mod_b[arg_b], color='orange', label="Planet b Model")
ax.legend(fontsize=12)
ax.set_ylabel("De-trended Flux [ppt]")
ax.set_xlabel("Phase")
ax.set_xlim(0, 1)
txt = "Planet b Period = {:.3f}".format(periods[0])
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
##################### PART 3: Build Model with Batman #################

# Could frist split them up into parts and use 'find min'-like function as a first guess
#troughs, trough_info = find_peaks(-flux_Kepler_masked, prominence = -0.001, width = 20)

params = batman.TransitParams()
params.t0 = t0is[0]                      #time of inferior conjunction
params.per = periods[0]                    #orbital period
params.rp = radii[0]                       #planet radius (in units of stellar radii)
params.a = a_array[0]/0.00465047*R_star[0] #semi-major axis (in units of stellar radii)
params.inc = incls[0]                      #orbital inclination (in degrees)
params.ecc = eccs[0]                       #eccentricity
params.w = 357.0                           #longitude of periastron (in degrees)
params.limb_dark = "nonlinear"             #limb darkening model
params.u = [0.5, 0.1, 0.1, -0.1]           #limb darkening coefficients [u1, u2, u3, u4]

t = time_Kepler_masked                     #times at which to calculate light curve
m = batman.TransitModel(params, t)
flux = m.light_curve(params) 

plt.figure()
plt.scatter(time_Kepler_masked, flux_Kepler_masked, c='k', s=1, label="Kepler De-trended Data")
plt.plot(time_Kepler_masked,flux)
#plt.plot(time_Kepler_masked[troughs], flux_Kepler_masked[troughs], "x", c = 'r')


# Calculate theoretical transit duration... Or simply read off model/data
    
############ PART 4: Minimize chi-squared in vicinity of each transit #########

# Define segments... T0 + n*period (+/- enough to catch)

#Work out number of transits
current_t0 = params.t0
initial_t0is = []

while current_t0 < time_Kepler[-1]:
    inital_t0is = initial_t0is.append(current_t0)
    current_t0 = current_t0 + periods[0]

n_list = range(len(initial_t0is))

final_t0is = [1]*len(n_list)
final_t0i_cs = [1]*len(n_list)

num_transits_done = 0

for n in n_list:
    int_start = t0is[0] + n*periods[0] - 2
    int_end = t0is[0] + n*periods[0] + 2
    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
    
    int_time = time_Kepler_masked[idx]
    int_flux = flux_Kepler_masked[idx]
    
    t0i_list = np.arange(int_start+1,int_end-1,0.05)
    
    chi_sq_list = []
    test_t0is = []    
    
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
        print('Number of transits analysed = {}'.format(num_transits_done))
        num_transits_done += 1

final_t0is = np.array(final_t0is)
initial_t0is = np.array(initial_t0is)

o_c = final_t0is - initial_t0is

plt.figure()
plt.scatter(time_Kepler_masked, flux_Kepler_masked, c='k', s=1, label="Kepler De-trended Data")
for t0i in initial_t0is:
    plt.axvline(x=t0i)
for t0i in final_t0is:
    plt.axvline(x=t0i, c='r')

 # find min chi-squared value out of all of them for each transit

#Plot new fold
folded_fig = plt.figure()
for n in n_list:
    int_start = final_t0is[n] - 2
    int_end = final_t0is[n] + 2
    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
    
    int_time = time_Kepler_masked[idx] - final_t0is[n]
    int_flux = flux_Kepler_masked[idx] 
    plt.scatter(int_time, int_flux, s=1, c='k')
plt.title('Folded fig for Kepler-9b')
plt.xlabel('Time since transit [Days]')
plt.ylabel('Normalized Flux')
    

final_t0is2, o_c2 = find_ttvs(final_t0is, periods[0], time_Kepler_masked, flux_Kepler_masked, params)

#Plot new fold
folded_fig2 = plt.figure()
for n in n_list:
    int_start = final_t0is2[n] - 2
    int_end = final_t0is2[n] + 2
    idx = np.where((time_Kepler_masked > int_start) & (time_Kepler_masked < int_end))
    
    int_time = time_Kepler_masked[idx] - final_t0is2[n]
    int_flux = flux_Kepler_masked[idx] 
    plt.scatter(int_time, int_flux, s=1, c='k')
plt.title('Folded fig for Kepler-9b - 2nd run')
plt.xlabel('Time since transit [Days]')
plt.ylabel('Normalized Flux')

# repeat model fit with all on top of each other and then repeat whole process
     # find new values for each one 
#     
#def build_model(mask=None, start=None, optimisation=True):
#    if mask is None:
#        mask = np.ones(len(time_TESS), dtype=bool)
#    with pm.Model() as model:
#
#		############################################################        
#		
#        ### Stellar parameters
#
#        mean = pm.Normal("mean", mu=0.0, sd=1.0)				#mean = the baseline flux = 0 for the TESS data 
#		# you can define a new mean for each set of photometry, just to keep track of it all
#		
#        u_star = xo.distributions.QuadLimbDark("u_star")				#kipping13 quad limb darkening
#		
#        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)			#using a bounded normal places constraints the prob distribution by introducing limits
#        m_star = BoundedNormal("m_star", mu=M_star[0], sd=M_star[1],testval=np.around(M_star[0], decimals = 1))	#stellar mass
#        r_star = BoundedNormal("r_star", mu=R_star[0], sd=R_star[1],testval=np.around(M_star[0], decimals = 1))	#stellar radius
#
#		############################################################    
#		
#        ### Orbital parameters for the planets
#        # Planet b
#        P_b = pm.Normal("P_b", mu=periods[0], sd=period_sds[0]) #the period (unlogged)
#        t0_b = pm.Normal("t0_b", mu=t0is[0], sd=t0i_sds[0])	#time of a ref transit for each planet
#        logr_b = pm.Normal("logr_b", mu=np.log(1 * radii[0]), sd=1.0)#log radius - we keep this one as a log
#        r_pl_b = pm.Deterministic("r_pl_b", tt.exp(logr_b)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
#        ratio_b = pm.Deterministic("ror_b", r_pl_b / r_star) #ratio - radius ratio between planet and star    		
#        b_b = xo.distributions.ImpactParameter("b_b", ror=ratio_b) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
#        #ecc_b = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
#        ecc_b = 0.0
#        #omega_b = xo.distributions.Angle("omega_b")
#        omega_b = 0.0            
#                
#        ############################################################    
#	
#		### Transit jitter & GP parameters for TESS LIGHTCURVE
#	
#        logs2 = pm.Normal("logs2", mu=np.log(np.var(flux_Kepler_masked)), sd=0.05)
#        logw0 = pm.Normal("logw0", mu=0.0, sd=0.05)
#        logSw4 = pm.Normal("logSw4", mu=np.log(np.var(flux_Kepler_masked)), sd=0.05)
#		# this sets up a GP for the TESS lightcurve, as per the tutorials.
#		# reducing sd seems to make the GP a little less wiggly
#        
#        ############################################################    
#
#		### Orbit model (Keplerian)
#        if method == 'manual':
#		#planet b
#            orbit_b = xo.orbits.KeplerianOrbit(
#                    r_star=r_star, 
#                    m_star=m_star,
#                    period=P_b,
#                    t0=t0_b,
#                    b=b_b,
#                    ecc=ecc_b,
#                    omega=omega_b,
#                    )
#    		#planet c
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
#        else:
#            # Planet b
#            orbit_b = xo.orbits.TTVOrbit(
#                    transit_times = final_t0is,
#                    r_star=r_star, 
#                    m_star=m_star,
#                    period=P_b,
#                    t0=t0_b,
#                    b=b_b,
#                    ecc=ecc_b,
#                    omega=omega_b,
#                    )
#	
#		############################################################            
#        ### Compute the model light curve using starry FOR TESS LIGHTCURVE
#		# it seems to break without the *1. 
#        
#
#        #planet b		
#        light_curves_b = (
#                xo.LimbDarkLightCurve(u_star).get_light_curve(
#                orbit=orbit_b, r=r_pl_b, t=time_Kepler_masked, texp=texp_TESS
#                )
#                * 1
#        )
#        light_curve_b = pm.math.sum(light_curves_b, axis=-1) + mean 	#this is the eclipse_model
#        pm.Deterministic("light_curves_b", light_curves_b) 			#tracking val of model light curve for plots
#        
#        if planet_num == 1:
#            light_curve_TESS = pm.math.sum(light_curves_b, axis=-1) + mean
#		
#		############################################################    
#		
#        ### GP model for the light curve
#		# Essentially from the tutorial
#	
#        kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2)) # n.b. SHOTerm = Stochastically driven, damped Harmonic Osciallator. Other recommended options are Matern32Term (Matern-3/2 function) and RotationTerm (two SHO for stellar rotation)
#        gp = xo.gp.GP(kernel, time_Kepler_masked, tt.exp(logs2) + tt.zeros(mask.sum()))
##        print(flux_TESS[mask])
##        print(light_curve_TESS)
#        pm.Potential("transit_obs", gp.log_likelihood(flux_Kepler_masked - light_curve_TESS))
#        pm.Deterministic("gp_pred", gp.predict())
#		
#	
#		### FITTING SEQUENCE		
#		
#        if start is None:
#            start = model.test_point
#        map_soln = xo.optimize(start=start)
#        map_soln = xo.optimize(start=map_soln)
#        map_soln = xo.optimize(start=map_soln)
#		
#        trace = []
#
#    return trace, model, map_soln, #vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H  # with RVs, you need to return some extra stuff for plotting
#
#trace, model0, map_soln0 = build_model() # this allows you to reuse the model or something later on - for GPs, add in: vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H
#
#print(map_soln0)          
#
#
#def plot_light_curve(soln, mask=None):
#    if mask is None:
#        mask = np.ones(len(time_Kepler_masked), dtype=bool)
#
#    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
#	
#    # this plot shows the og lightcurves with the gp model on top
#    ax = axes[0]
#    ax.scatter(time_Kepler_masked[mask], flux_Kepler_masked[mask], c='k', s = 1, label="Original Data")
#    gp_mod = soln["gp_pred"] + soln["mean"]
#    ax.plot(time_Kepler_masked[mask], gp_mod, color="C2", label="GP Model")
#    ax.legend(fontsize=12)
#    ax.set_ylabel("Relative Flux [ppt]")
#
#    # this plot shows the clean lightcurve (og lightcurve - gp solution) plus the light curve models for planets b and c
#    ax = axes[1]
#    ax.scatter(time_Kepler_masked[mask], flux_Kepler_masked[mask] - gp_mod, c='k', s=1, label="De-trended Data")
#    mod_b = soln["light_curves_b"]
#    ax.plot(time_Kepler_masked[mask], mod_b, color='orange', label="Planet b Model")
#    ax.legend(fontsize=12)
#    ax.set_ylabel("De-trended Flux [ppt]")
#	
#    # this plot shows the residuals
#    ax = axes[2]
#    mod = gp_mod + np.sum(soln["light_curves_b"], axis=-1)
#    ax.scatter(time_Kepler_masked[mask], flux_Kepler_masked[mask] - mod, c='k', s=1, label='Residuals')
#    ax.axhline(0, color="mediumvioletred", lw=1, label = 'Baseline Flux')
#    ax.set_ylabel("Residuals [ppt]")
#    ax.legend(fontsize=12)
#    ax.set_xlim(time_Kepler_masked[mask].min(), time_Kepler_masked[mask].max())
#    ax.set_xlabel("Time [BJD - 2454833]")
#    plt.title('{}'.format(target_ID))
#	
#    plt.subplots_adjust(hspace=0)
#
#    return fig
#
#plot_light_curve(map_soln0)

#################### PART 5: Compute and plot O-C diagram #####################
# Using final observed values compute O-C values for all transits and plot
o_c_hrs = o_c*24     
     
o_c_plot = plt.figure()
plt.scatter(final_t0is2, o_c2, c='k',s=1)
plt.xlabel('BJD Time [Days]')
plt.ylabel('O-C [hrs]')
plt.title('O-C diagram for Kepler-9b')
