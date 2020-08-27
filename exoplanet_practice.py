#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:28:07 2020

Exoplanet practice

@author: mbattley
"""

import sys
import corner
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
from astropy.table import Table
from lc_download_methods import two_min_lc_download
from remove_tess_systematics import clean_tess_lc
from astropy.io import fits
from astropy.time import Time
from astropy import coordinates as coord
from astropy import constants as const
from exoplanet.gp import terms
from lightkurve import search_lightcurvefile

#def my_custom_corrector_func(lc):
#    corrected_lc = lc.normalize().flatten(window_length=401)
#    return corrected_lc
def binned(time, flux, binsize=15, method='mean'):
    """Bins a lightcurve in blocks of size `binsize`.
    n.b. based on the one from eleanor
    """
    available_methods = ['mean', 'median']
    if method not in available_methods:
        raise ValueError("method must be one of: {}".format(available_methods))
    methodf = np.__dict__['nan' + method]

    n_bins = len(flux) // binsize
    indexes = np.array_split(np.arange(len(time)), n_bins)
    binned_time = np.array([methodf(time[a]) for a in indexes])
    binned_flux = np.array([methodf(flux[a]) for a in indexes])

    return binned_time, binned_flux

############################## PART 0: Setup ##################################
# lc parameters
save_path = '/Users/mbattley/Documents/PhD/Kepler-2min xmatch/'
target_ID = 'TIC 27769688' # Kepler 25 test-case: 'TIC 120960812' #TIC number
TIC = int(target_ID[4:])
sector = 14
multi_sector = False
planet_data = Table.read(save_path + 'Kepler_planets_reobserved_in_TESS_2min.csv', format='ascii.csv')
#target_ID_list = np.array(pc_data['TICID'])
i = list(planet_data['TICID']).index(int(target_ID[3:]))
instrument = 'both'
planet_letter = ['b','c','d','e','f','g','h']
method = 'auto' #Can be 'array, 'auto' or 'manual'

################### PART 1: Downloading/Opening Light-curves ##################
# TESS 2min
if (planet_data['S14'][i] != 0) and (planet_data['S15'][i] != 0):
    multi_sector = [14,15]
#multi_sector = False
if multi_sector != False:
    sap_lc, pdcsap_lc = two_min_lc_download(TIC, sector = multi_sector[0], from_file = False)
    lc = pdcsap_lc
    nancut = np.isnan(lc.flux) | np.isnan(lc.time)
    lc = lc[~nancut]
    for sector_num in multi_sector[1:]:
        sap_lc_new, pdcsap_lc_new = two_min_lc_download(TIC, sector_num, from_file = False)
        lc_new = pdcsap_lc_new
        nancut = np.isnan(lc_new.flux) | np.isnan(lc_new.time)
        lc_new = lc_new[~nancut]
        lc = lc.append(lc_new)
else:
#    sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = sector, from_file = False)
    lcf = search_lightcurvefile(target_ID, sector=sector).download()
    pdcsap_lc = lcf.PDCSAP_FLUX
    header_0 = lcf
    lc = pdcsap_lc
    nancut = np.isnan(lc.flux) | np.isnan(lc.time)
    lc = lc[~nancut]
    print('Removed nans')
time_TESS = np.array(lc.time) #n.b. in TJD (TESS Time)
time_TESS_orig = np.array([float(str(element).strip()) for element in time_TESS]) + 2457000 #Convert to BJD for consistency
flux_TESS = lc.flux
flux_TESS_orig = np.array(flux_TESS)/np.median(flux_TESS) -1 #Normalizes and sets mean to zero, as in exoplanet tutorial

# Kepler
KIC = planet_data['KIC'][i]
lcfs = lightkurve.search_lightcurvefile(KIC, mission='Kepler').download_all()
stitched_lc = lcfs.PDCSAP_FLUX.stitch()
#stitched_lc = lcfs.PDCSAP_FLUX.stitch(corrector_func=my_custom_corrector_func)
nancut = np.isnan(stitched_lc.flux) | np.isnan(stitched_lc.time)
stitched_lc = stitched_lc[~nancut]
time_Kepler = np.array(stitched_lc.time) + 2454833 #Convert to BJD for consistency
flux_Kepler = np.array(stitched_lc.flux)/np.median(stitched_lc.flux) -1
binned_Kepler_time, binned_Kepler_flux = binned(time_Kepler,flux_Kepler,binsize=2)
#yerr = stitched_lc.flux_err

if instrument == 'TESS':
    time_TESS = time_TESS_orig
    flux_TESS = flux_TESS_orig
elif instrument == 'Kepler':
    time_TESS = time_Kepler
    flux_TESS = flux_Kepler
elif instrument == 'both':
    time_TESS = np.append(time_Kepler,time_TESS_orig)
    flux_TESS = np.append(flux_Kepler,flux_TESS_orig)

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
    planet_params = Table(names=('periodi','periodi_sd','t0i','t0i_sd','radi'))
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
            planet_params.add_row((periodi, period_sd, t0i, t0i_sd, radi))
            planet_num += 1
    periods = np.array(planet_params['periodi'])
    period_sds = np.array(planet_params['periodi_sd'])
    t0is = np.array(planet_params['t0i'])
    t0i_sds = np.array(planet_params['t0i_sd'])
#    t0i_sds = np.array([0.1]*planet_num)
    radii = np.array(planet_params['radi'])
#    planet_num = 4
#method = 'manual'

####################### PART 3: Model Construction ############################

def build_model(mask=None, start=None, optimisation=True):
    if mask is None:
        mask = np.ones(len(time_TESS), dtype=bool)
    with pm.Model() as model:

		############################################################        
		
        ### Stellar parameters

        mean = pm.Normal("mean", mu=0.0, sd=1.0)				#mean = the baseline flux = 0 for the TESS data 
		# you can define a new mean for each set of photometry, just to keep track of it all
		
        u_star = xo.distributions.QuadLimbDark("u_star")				#kipping13 quad limb darkening
		
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)			#using a bounded normal places constraints the prob distribution by introducing limits
        m_star = BoundedNormal("m_star", mu=M_star[0], sd=M_star[1],testval=np.around(M_star[0], decimals = 1))	#stellar mass
        r_star = BoundedNormal("r_star", mu=R_star[0], sd=R_star[1],testval=np.around(M_star[0], decimals = 1))	#stellar radius

		############################################################    
		
        ### Orbital parameters for the planets
        
        if method == 'manual':
            P_b = pm.Normal("P_b", mu=periodi_b, sd=periodi_sd_b) #the period (unlogged)
            P_c = pm.Normal("P_c", mu=periodi_c, sd=periodi_sd_c) 
    	   
            t0_b = pm.Normal("t0_b", mu=t0i_b, sd=t0i_sd_b)	#time of a ref transit for each planet
            t0_c = pm.Normal("t0_c", mu=t0i_c, sd=t0i_sd_c)
    		
        		#log radius - we keep this one as a log
            logr_b = pm.Normal("logr_b", mu=np.log(1 * radi_b), sd=1.0)
            logr_c = pm.Normal("logr_c", mu= np.log(1 * radi_c), sd=1.0)
    		
        		#radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!
            r_pl_b = pm.Deterministic("r_pl_b", tt.exp(logr_b))
            r_pl_c = pm.Deterministic("r_pl_c", tt.exp(logr_c))		
    		
        		#ratio - radius ratio between planet and star
            ratio_b = pm.Deterministic("ror_b", r_pl_b / r_star) 
            ratio_c = pm.Deterministic("ror_c", r_pl_c / r_star)
    		
            b_b = xo.distributions.ImpactParameter("b_b", ror=ratio_b) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
            b_c = xo.distributions.ImpactParameter("b_c", ror=ratio_c)
            
            #ecc_b = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
            ecc_b = 0.0
#            omega_b = xo.distributions.Angle("omega_b")
            omega_b = 0.0
		
        		#ecc_c = xo.distributions.UnitUniform("ecc_c", testval=0.0)
            ecc_c = 0.0
#            c = xo.distributions.Angle("omega_c")
            omega_c = 0.0
        elif method == 'array':
#            logP = pm.Normal("logP", mu=np.log(periods), sd=0.1, shape=planet_num)
            P = pm.Normal("period", mu=periods, sd=period_sds, shape = planet_num)
#            P = pm.Deterministic("period", pm.math.exp(logP))
            t0 = pm.Normal("t0", mu=t0is, sd=t0i_sds, shape = planet_num)
            logr = pm.Normal("logr", mu=radii, sd=1.0, shape = planet_num)
            r_pl = pm.Deterministic("r_pl", tt.exp(logr))
            ratio = pm.Deterministic("ror", r_pl / r_star)
            b = xo.distributions.ImpactParameter("b", ror=ratio,shape=planet_num, testval=np.random.rand(planet_num))
            ecc = [0]*planet_num
            omega = [0]*planet_num
        else:
            # Planet b
            P_b = pm.Normal("P_b", mu=periods[0], sd=period_sds[0]) #the period (unlogged)
            t0_b = pm.Normal("t0_b", mu=t0is[0], sd=t0i_sds[0])	#time of a ref transit for each planet
            logr_b = pm.Normal("logr_b", mu=np.log(1 * radii[0]), sd=1.0)#log radius - we keep this one as a log
            r_pl_b = pm.Deterministic("r_pl_b", tt.exp(logr_b)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
            ratio_b = pm.Deterministic("ror_b", r_pl_b / r_star) #ratio - radius ratio between planet and star    		
            b_b = xo.distributions.ImpactParameter("b_b", ror=ratio_b) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
            #ecc_b = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
            ecc_b = 0.0
#            omega_b = xo.distributions.Angle("omega_b")
            omega_b = 0.0
            
            # Planet c (where appropriate)
            if planet_num > 1:
                P_c = pm.Normal("P_c", mu=periods[1], sd=period_sds[1]) #the period (unlogged)
                t0_c = pm.Normal("t0_c", mu=t0is[1], sd=t0i_sds[1])	#time of a ref transit for each planet
                logr_c = pm.Normal("logr_c", mu=np.log(1 * radii[1]), sd=1.0)#log radius - we keep this one as a log
                r_pl_c = pm.Deterministic("r_pl_c", tt.exp(logr_c)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
                ratio_c = pm.Deterministic("ror_c", r_pl_c / r_star) #ratio - radius ratio between planet and star    		
                b_c = xo.distributions.ImpactParameter("b_c", ror=ratio_c) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
                #ecc_c = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
                ecc_c = 0.0
                #omega_c = xo.distributions.Angle("omega_b")
                omega_c = 0.0
            
            # Planet d (where appropriate)
            if planet_num > 2:
                P_d = pm.Normal("P_d", mu=periods[2], sd=period_sds[2]) #the period (unlogged)
                t0_d = pm.Normal("t0_d", mu=t0is[2], sd=t0i_sds[2])	#time of a ref transit for each planet
                logr_d = pm.Normal("logr_d", mu=np.log(1 * radii[2]), sd=1.0)#log radius - we keep this one as a log
                r_pl_d = pm.Deterministic("r_pl_d", tt.exp(logr_d)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
                ratio_d = pm.Deterministic("ror_d", r_pl_d / r_star) #ratio - radius ratio between planet and star    		
                b_d = xo.distributions.ImpactParameter("b_d", ror=ratio_d) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
                #ecc_c = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
                ecc_d = 0.0
                #omega_c = xo.distributions.Angle("omega_b")
                omega_d = 0.0
            
            # Planet e (where appropriate)
            if planet_num > 3:
                P_e = pm.Normal("P_e", mu=periods[3], sd=period_sds[3]) #the period (unlogged)
                t0_e = pm.Normal("t0_e", mu=t0is[3], sd=t0i_sds[3])	#time of a ref transit for each planet
                logr_e = pm.Normal("logr_e", mu=np.log(1 * radii[3]), sd=1.0)#log radius - we keep this one as a log
                r_pl_e = pm.Deterministic("r_pl_e", tt.exp(logr_e)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
                ratio_e = pm.Deterministic("ror_e", r_pl_e / r_star) #ratio - radius ratio between planet and star    		
                b_e = xo.distributions.ImpactParameter("b_e", ror=ratio_e) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
                #ecc_c = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
                ecc_e = 0.0
                #omega_c = xo.distributions.Angle("omega_b")
                omega_e = 0.0
            
            # Planet f (where appropriate)
            if planet_num > 4:
                P_f = pm.Normal("P_f", mu=periods[4], sd=period_sds[4]) #the period (unlogged)
                t0_f = pm.Normal("t0_f", mu=t0is[4], sd=t0i_sds[4])	#time of a ref transit for each planet
                logr_f = pm.Normal("logr_f", mu=np.log(1 * radii[4]), sd=1.0)#log radius - we keep this one as a log
                r_pl_f = pm.Deterministic("r_pl_f", tt.exp(logr_f)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
                ratio_f = pm.Deterministic("ror_f", r_pl_f / r_star) #ratio - radius ratio between planet and star    		
                b_f = xo.distributions.ImpactParameter("b_f", ror=ratio_f) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
                #ecc_c = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
                ecc_f = 0.0
                #omega_c = xo.distributions.Angle("omega_b")
                omega_f = 0.0
               
            # Planet g (where appropriate)
            if planet_num > 5:
                P_g = pm.Normal("P_g", mu=periods[5], sd=period_sds[5]) #the period (unlogged)
                t0_g = pm.Normal("t0_g", mu=t0is[5], sd=t0i_sds[5])	#time of a ref transit for each planet
                logr_g = pm.Normal("logr_g", mu=np.log(1 * radii[5]), sd=1.0)#log radius - we keep this one as a log
                r_pl_g = pm.Deterministic("r_pl_g", tt.exp(logr_g)) #radius - we then unlog the radius to keep track of it. a pm.Deterministic basically just tracks a value for you for later on!	
                ratio_g = pm.Deterministic("ror_g", r_pl_g / r_star) #ratio - radius ratio between planet and star    		
                b_g = xo.distributions.ImpactParameter("b_g", ror=ratio_g) # we used the xo distribution rather than the pymc3 one for b, as per the tutorial
                #ecc_c = xo.distributions.UnitUniform("ecc_b", testval=0.0) # at the moment, we're fixing e to 0 for simplicity.
                ecc_g = 0.0
                #omega_c = xo.distributions.Angle("omega_b")
                omega_g = 0.0
                
        ############################################################    
	
		### Transit jitter & GP parameters for TESS LIGHTCURVE
	
        logs2 = pm.Normal("logs2", mu=np.log(np.var(flux_TESS[mask])), sd=0.05)
        logw0 = pm.Normal("logw0", mu=0.0, sd=0.05)
        logSw4 = pm.Normal("logSw4", mu=np.log(np.var(flux_TESS[mask])), sd=0.05)
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
                    b=b_b,
                    ecc=ecc_b,
                    omega=omega_b,
                    )
    		#planet c
            orbit_c = xo.orbits.KeplerianOrbit(
                    r_star=r_star,
                    m_star=m_star,
                    period=P_c,
                    t0=t0_c,
                    b=b_c,
                    ecc=ecc_c,
                    omega=omega_c,
                    )
        elif method == 'array':
            orbit = xo.orbits.KeplerianOrbit(                    
                    r_star=r_star,
                    m_star=m_star,
                    period=P,
                    t0=t0,
                    b=b,
                    ecc=ecc,
                    omega=omega)
        else:
            # Planet b
            orbit_b = xo.orbits.KeplerianOrbit(
                    r_star=r_star, 
                    m_star=m_star,
                    period=P_b,
                    t0=t0_b,
                    b=b_b,
                    ecc=ecc_b,
                    omega=omega_b,
                    )
            # Planet c
            if planet_num > 1:
                orbit_c = xo.orbits.KeplerianOrbit(
                        r_star=r_star, 
                        m_star=m_star,
                        period=P_c,
                        t0=t0_c,
                        b=b_c,
                        ecc=ecc_c,
                        omega=omega_c,
                        )
            # Planet d
            if planet_num > 2:
                orbit_d = xo.orbits.KeplerianOrbit(
                        r_star=r_star, 
                        m_star=m_star,
                        period=P_d,
                        t0=t0_d,
                        b=b_d,
                        ecc=ecc_d,
                        omega=omega_d,
                        )
            
            # Planet e
            if planet_num > 3:
                orbit_e = xo.orbits.KeplerianOrbit(
                        r_star=r_star, 
                        m_star=m_star,
                        period=P_e,
                        t0=t0_e,
                        b=b_e,
                        ecc=ecc_e,
                        omega=omega_e,
                        )

            # Planet f
            if planet_num > 4:
                orbit_f = xo.orbits.KeplerianOrbit(
                        r_star=r_star, 
                        m_star=m_star,
                        period=P_f,
                        t0=t0_f,
                        b=b_f,
                        ecc=ecc_f,
                        omega=omega_f,
                        )
                
            # Planet g
            if planet_num > 5:
                orbit_g = xo.orbits.KeplerianOrbit(
                        r_star=r_star, 
                        m_star=m_star,
                        period=P_g,
                        t0=t0_g,
                        b=b_g,
                        ecc=ecc_g,
                        omega=omega_g,
                        )
	
		############################################################            
        ### Compute the model light curve using starry FOR TESS LIGHTCURVE
		# it seems to break without the *1. 
        
        if method == 'manual':
            #planet b		
            light_curves_b = (
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                    orbit=orbit_b, r=r_pl_b, t=time_TESS[mask], texp=texp_TESS
                    )
                    * 1
            )
            light_curve_b = pm.math.sum(light_curves_b, axis=-1) + mean 	#this is the eclipse_model
            pm.Deterministic("light_curves_b", light_curves_b) 			#tracking val of model light curve for plots
		
		#planet c - same thing as above
            light_curves_c = (
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                    orbit=orbit_c, r=r_pl_c, t=time_TESS[mask], texp=texp_TESS
                    )
                    * 1
            )
            light_curve_c = pm.math.sum(light_curves_c, axis=-1) + mean
            pm.Deterministic("light_curves_c", light_curves_c)
            
            light_curve_TESS = pm.math.sum(light_curves_b+light_curves_c, axis=-1) + mean
        
        elif method == 'array':
            light_curves = (
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                    orbit=orbit, r=r_pl, t=time_TESS[mask], texp=texp_TESS
                    )
                    * 1
            )
            light_curve = pm.math.sum(light_curves, axis=-1) + mean
            pm.Deterministic("light_curves", light_curves)
        
        else:
            #planet b		
            light_curves_b = (
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                    orbit=orbit_b, r=r_pl_b, t=time_TESS[mask], texp=texp_TESS
                    )
                    * 1
            )
            light_curve_b = pm.math.sum(light_curves_b, axis=-1) + mean 	#this is the eclipse_model
            pm.Deterministic("light_curves_b", light_curves_b) 			#tracking val of model light curve for plots
            
            #planet c
            if planet_num >1:
                light_curves_c = (
                        xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit_c, r=r_pl_c, t=time_TESS[mask], texp=texp_TESS
                        )
                        * 1
                )
                light_curve_c = pm.math.sum(light_curves_c, axis=-1) + mean
                pm.Deterministic("light_curves_c", light_curves_c)
            
            #planet d
            if planet_num >2:
                light_curves_d = (
                        xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit_d, r=r_pl_d, t=time_TESS[mask], texp=texp_TESS
                        )
                        * 1
                )
                light_curve_d = pm.math.sum(light_curves_d, axis=-1) + mean
                pm.Deterministic("light_curves_d", light_curves_d)
            
            #planet e
            if planet_num >3:
                light_curves_e = (
                        xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit_e, r=r_pl_e, t=time_TESS[mask], texp=texp_TESS
                        )
                        * 1
                )
                light_curve_e = pm.math.sum(light_curves_e, axis=-1) + mean
                pm.Deterministic("light_curves_e", light_curves_e)
            
            #planet f
            if planet_num >4:
                light_curves_f = (
                        xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit_f, r=r_pl_f, t=time_TESS[mask], texp=texp_TESS
                        )
                        * 1
                )
                light_curve_f = pm.math.sum(light_curves_f, axis=-1) + mean
                pm.Deterministic("light_curves_f", light_curves_f)
            
            #planet g
            if planet_num >5:
                light_curves_g = (
                        xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit_g, r=r_pl_g, t=time_TESS[mask], texp=texp_TESS
                        )
                        * 1
                )
                light_curve_g = pm.math.sum(light_curves_g, axis=-1) + mean
                pm.Deterministic("light_curves_g", light_curves_g)
            
            if planet_num == 1:
                light_curve_TESS = pm.math.sum(light_curves_b, axis=-1) + mean
            elif planet_num == 2:
                light_curve_TESS = pm.math.sum(light_curves_b+light_curves_c, axis=-1) + mean
            elif planet_num == 3:
                light_curve_TESS = pm.math.sum(light_curves_b+light_curves_c+light_curves_d, axis=-1) + mean
            elif planet_num == 4:
                light_curve_TESS = pm.math.sum(light_curves_b+light_curves_c+light_curves_d+light_curves_e, axis=-1) + mean
            elif planet_num == 5:
                light_curve_TESS = pm.math.sum(light_curves_b+light_curves_c+light_curves_d+light_curves_e+light_curves_f, axis=-1) + mean
            elif planet_num == 6:
                light_curve_TESS = pm.math.sum(light_curves_b+light_curves_c+light_curves_d+light_curves_e+light_curves_f+light_curves_g, axis=-1) + mean
		
		############################################################    
		
        ### GP model for the light curve
		# Essentially from the tutorial
	
        kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2)) # n.b. SHOTerm = Stochastically driven, damped Harmonic Osciallator. Other recommended options are Matern32Term (Matern-3/2 function) and RotationTerm (two SHO for stellar rotation)
        gp = xo.gp.GP(kernel, time_TESS[mask], tt.exp(logs2) + tt.zeros(mask.sum()))
#        print(flux_TESS[mask])
#        print(light_curve_TESS)
        pm.Potential("transit_obs", gp.log_likelihood(flux_TESS[mask] - light_curve_TESS))
        pm.Deterministic("gp_pred", gp.predict())
		
		############################################################
	
		### RV SHENANIGANS
		
#		vrad_b = orbit_b.get_radial_velocity(HARPS_RV_time, K=tt.exp(logK_b)) # you gotta define a vrad for each planet
#		vrad_c = orbit_c.get_radial_velocity(HARPS_RV_time, K=tt.exp(logK_c))
#		vrad_d = orbit_d.get_radial_velocity(HARPS_RV_time, K=tt.exp(logK_d))
#		pm.Deterministic("vrad_b", vrad_b)
#		pm.Deterministic("vrad_c", vrad_c)
#		pm.Deterministic("vrad_d", vrad_d)
#		
#		
#		# okay, this is my own GP created from pymc3 stuff. so ignore this, really, until you come to GPing.
#		# exoplanet doesn't have it's own quasiperiodic kernel, so i created my own.
#		# but it's currently being tested
#		period_H = pm.Normal("period", mu=30.0, sd=10.0)
#		amp = pm.HalfCauchy("eta", beta=5) 								# amplitude term
#		ls_EQ = pm.TruncatedNormal("lengthscaleExp", mu=8.0, sigma=10.0, lower=0)		# lengthscale for ExpQuad
#		ls_Per = pm.TruncatedNormal("lengthscalePer", mu=30.0, sigma=10.0, lower=0)	# lengthscale for Periodic
#
#		expquadgp = pm.gp.cov.ExpQuad(1, ls=ls_EQ) # ExpQuad kernel/covariance func
#		periodicgp = pm.gp.cov.Periodic(1, period_H, ls=ls_Per) # Periodic kernel/covariance func
#		
#		QuasiPer = amp**2 * expquadgp * periodicgp # full Quasi Periodic kernel
#		
#		gp_H = pm.gp.Marginal(cov_func=QuasiPer)
#		
#		HARPS_offset = pm.Normal("HARPS_offset", mu=48830.87, sigma=10)
#		
#		mean_H = vrad_b + vrad_c + vrad_d + HARPS_offset
#		jitter2_H = pm.Normal('log_HARPS_jitter2', mu=2.*np.log(np.min(HARPS_RV_vraderr)), sd=5.0)
#		
#		HARPS_RV_time_reshaped = HARPS_RV_time.reshape(len(HARPS_RV_time),1)
#		
#		X = HARPS_RV_time_reshaped
#		y = HARPS_RV_vrad - mean_H
#
#		out = gp_H.marginal_likelihood("out", X=X, y=y, noise=pm.math.exp(jitter2_H) + HARPS_RV_vraderr ** 2)	
		
		############################################################
	
		### FITTING SEQUENCE		
		
        if start is None:
            start = model.test_point
        map_soln = xo.optimize(start=start)
        map_soln = xo.optimize(start=map_soln)
        map_soln = xo.optimize(start=map_soln)
		
#		finegrid = np.linspace(HARPS_RV_time.min(), HARPS_RV_time.max(), num=5000)
#		finegrid_gp = np.linspace(HARPS_RV_time.min(), HARPS_RV_time.max(), num=5000)[:,None]

#		vrad_b_plot = xo.eval_in_model(orbit_b.get_radial_velocity(finegrid,K=np.exp(map_soln['logK_b'])),map_soln)
#		vrad_c_plot = xo.eval_in_model(orbit_c.get_radial_velocity(finegrid,K=np.exp(map_soln['logK_c'])),map_soln)
#		vrad_d_plot = xo.eval_in_model(orbit_d.get_radial_velocity(finegrid,K=np.exp(map_soln['logK_d'])),map_soln)
        trace = []
#        trace = pm.sample(
#                tune=1000,
#                draws=1000,
#                start=map_soln,
#                cores=2,
#                chains=2,
#                step=xo.get_dense_nuts_step(target_accept=0.9),
#        )
#        print(pm.stats.summary(trace, var_names=["P_b","t0_b", "r_pl_b"]))

#        samples = pm.trace_to_dataframe(trace, varnames=["P_b","t0_b", "r_pl_b"])
##        model.test_point = [periods[0],t0is[0],0.02]
#        truth = np.concatenate(
#            xo.eval_in_model([P_b, t0_b, r_pl_b], model=model)
#        )
#        _ = corner.corner(
#            samples,
#            truths=truth,
#            labels=["P_b","t0_b", "r_pl_b"],
#        )
    return trace, model, map_soln, #vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H  # with RVs, you need to return some extra stuff for plotting

trace, model0, map_soln0 = build_model() # this allows you to reuse the model or something later on - for GPs, add in: vrad_b_plot, vrad_c_plot, vrad_d_plot, gp_H

print(map_soln0) # this is how you find out the values your parameters have fit to! it prints a huge long thing in your terminal lmao

with open(save_path+'Fit_solutions/{}_fit_soln.csv'.format(target_ID),'w') as f:
    w = csv.writer(f)
    w.writerows(map_soln0.items())


############################# PART 4: Plotting ################################

# Setup
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams['xtick.top']=True
plt.rcParams['ytick.right']=True
#plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 12})

############# TESS lightcurve GP, model, residuals ############

# the ouput line plot_light_curve defines that soln is the map_soln_0
# in order to get a value for a parameter or model from within your overall model, you do soln['parametername'] and it grabs it for you

def plot_light_curve(soln, mask=None):
    if mask is None:
        mask = np.ones(len(time_TESS), dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
	
    # this plot shows the og lightcurves with the gp model on top
    ax = axes[0]
    ax.scatter(time_TESS[mask], flux_TESS[mask], c='k', s = 1, label="Original Data")
    gp_mod = soln["gp_pred"] + soln["mean"]
    ax.plot(time_TESS[mask], gp_mod, color="C2", label="GP Model")
    ax.legend(fontsize=12)
    ax.set_ylabel("Relative Flux [ppt]")

    # this plot shows the clean lightcurve (og lightcurve - gp solution) plus the light curve models for planets b and c
    ax = axes[1]
    ax.scatter(time_TESS[mask], flux_TESS[mask] - gp_mod, c='k', s=1, label="De-trended Data")
    mod_b = soln["light_curves_b"]
    ax.plot(time_TESS[mask], mod_b, color='orange', label="Planet b Model")
    mod_c = soln["light_curves_c"]
    ax.plot(time_TESS[mask], mod_c, color='blue', label="Planet c Model")
    ax.legend(fontsize=12)
    ax.set_ylabel("De-trended Flux [ppt]")
	
    # this plot shows the residuals
    ax = axes[2]
    mod = gp_mod + np.sum(soln["light_curves_b"] + soln["light_curves_c"], axis=-1)
    ax.scatter(time_TESS[mask], flux_TESS[mask] - mod, c='k', s=1, label='Residuals')
    ax.axhline(0, color="mediumvioletred", lw=1, label = 'Baseline Flux')
    ax.set_ylabel("Residuals [ppt]")
    ax.legend(fontsize=12)
    ax.set_xlim(time_TESS[mask].min(), time_TESS[mask].max())
    ax.set_xlabel("Time [BJD - 2454833]")
    plt.title('{}'.format(target_ID))
	
    plt.subplots_adjust(hspace=0)

    return fig

def plot_light_curve_auto(soln, mask=None):
    if mask is None:
        mask = np.ones(len(time_TESS), dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # this plot shows the og lightcurves with the gp model on top
    ax = axes[0]
    ax.scatter(time_TESS[mask], flux_TESS[mask], c='k', s = 1, label="Original Data")
    gp_mod = soln["gp_pred"] + soln["mean"]
    ax.plot(time_TESS[mask], gp_mod, color="C2", label="GP Model")
    ax.legend(fontsize=12)
    ax.set_ylabel("Relative Flux [ppt]")
    ax.set_title('{}'.format(target_ID))

    # this plot shows the clean lightcurve (og lightcurve - gp solution) plus the light curve models for planets b and c
    ax = axes[1]
    ax.scatter(time_TESS[mask], flux_TESS[mask] - gp_mod, c='k', s=1, label="De-trended Data")
    mod_b = soln["light_curves_b"]
    ax.plot(time_TESS[mask], mod_b, color='orange', label="Planet b Model")
    if planet_num > 1:
        mod_c = soln["light_curves_c"]
        ax.plot(time_TESS[mask], mod_c, color='blue', label="Planet c Model")
    if planet_num > 2:
        mod_d = soln["light_curves_d"]
        ax.plot(time_TESS[mask], mod_d, color='cyan', label="Planet d Model")
    if planet_num > 3:
        mod_e = soln["light_curves_e"]
        ax.plot(time_TESS[mask], mod_e, color='green', label="Planet e Model")
    if planet_num > 4:
        mod_f = soln["light_curves_f"]
        ax.plot(time_TESS[mask], mod_f, color='red', label="Planet f Model")
    if planet_num > 5:
        mod_g = soln["light_curves_g"]
        ax.plot(time_TESS[mask], mod_g, color='purple', label="Planet g Model")
    ax.legend(fontsize=12)
    ax.set_ylabel("De-trended Flux [ppt]")
	
    # this plot shows the residuals
    ax = axes[2]
    if planet_num == 1:
        mod = gp_mod + np.sum(soln["light_curves_b"], axis=-1)
    elif planet_num == 2:
        mod = gp_mod + np.sum(soln["light_curves_b"] + soln["light_curves_c"], axis=-1)
    elif planet_num == 3:
        mod = gp_mod + np.sum(soln["light_curves_b"] + soln["light_curves_c"] + soln["light_curves_d"], axis=-1)
    elif planet_num == 4:
        mod = gp_mod + np.sum(soln["light_curves_b"] + soln["light_curves_c"] + soln["light_curves_d"] + soln["light_curves_e"], axis=-1)
    elif planet_num == 5:
        mod = gp_mod + np.sum(soln["light_curves_b"] + soln["light_curves_c"] + soln["light_curves_d"] + soln["light_curves_e"] + soln["light_curves_f"], axis=-1)
    elif planet_num == 6:
        mod = gp_mod + np.sum(soln["light_curves_b"] + soln["light_curves_c"] + soln["light_curves_d"] + soln["light_curves_e"] + soln["light_curves_f"] + soln["light_curves_g"], axis=-1)
    ax.scatter(time_TESS[mask], flux_TESS[mask] - mod, c='k', s=1, label='Residuals')
    ax.axhline(0, color="mediumvioletred", lw=1, label = 'Baseline Flux')
    ax.set_ylabel("Residuals [ppt]")
    ax.legend(fontsize=12)
    ax.set_xlim(time_TESS[mask].min(), time_TESS[mask].max())
    ax.set_xlabel("Time [BJD]")
	
    plt.subplots_adjust(hspace=0)

    return fig

if method == 'manual':
    plot_light_curve(map_soln0);
else:
    plot_light_curve_auto(map_soln0);
#
#
############### TESS lightcurve phase folding, with model on top ###############
#
def plot_light_curve(soln, mask=None,instrument = 'both'):
    if instrument == 'both':
        if mask is None:
        		mask = np.ones(len(time_TESS), dtype=bool)
        
        gp_mod = soln["gp_pred"] + soln["mean"]
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    	
        # setting up the phase fold data
        phases_b = np.mod(time_TESS - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
        phases_b_TESS = np.mod(time_TESS_orig - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
        phases_b_Kepler = np.mod(time_Kepler - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
    	
        phases_c_TESS = np.mod(time_TESS_orig - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
        phases_c_Kepler = np.mod(time_Kepler - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
        phases_c = np.mod(time_TESS - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
    	
        arg_b = np.argsort(phases_b)
        arg_c = np.argsort(phases_c)
    	
        # phase fold for planet b
        ax = axes[0]
        ax.scatter(phases_b_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1, label="TESS De-trended Data")
        ax.scatter(phases_b_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1, label="Kepler De-trended Data")
        mod_b = soln["light_curves_b"]
        ax.plot(phases_b[mask][arg_b], mod_b[arg_b], color='orange', label="Planet b Model")
        ax.legend(fontsize=12)
        ax.set_ylabel("De-trended Flux [ppt]")
        ax.set_xlabel("Phase")
        ax.set_xlim(0, 1)
        txt = "Planet b Period = {:.3f}".format(map_soln0['P_b'])
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
        
        # phase fold for planet c
        ax = axes[1]
        ax.scatter(phases_c_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1, label="Kepler De-trended Data".format(instrument))
        ax.scatter(phases_c_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1, label="TESS De-trended Data".format(instrument))
        mod_c = soln["light_curves_c"]
        ax.plot(phases_c[mask][arg_c], mod_c[arg_c], color='blue', label="Planet c Model")
        ax.legend(fontsize=12)
        ax.set_ylabel("De-trended Flux [ppt]")
        ax.set_xlim(0, 1)
        txt = "Planet c Period = {:.3f}".format(map_soln0['P_c'])
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
            
    else:
        fig, axes = plt.subplots(planet_num, 1, figsize=(8, 10), sharex=False)
    	
        # setting up the phase fold data
        phases_c = np.mod(time_TESS - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
    	
        phases_b = np.mod(time_TESS - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
    	
        arg_c = np.argsort(phases_c)
        arg_b = np.argsort(phases_b)
    	
        # phase fold for planet b
        ax = axes[0]
        ax.scatter(phases_b[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
        mod_b = soln["light_curves_b"]
        ax.plot(phases_b[mask][arg_b], mod_b[arg_b], color='orange', label="Planet b Model")
        ax.legend(fontsize=12)
        ax.set_ylabel("De-trended Flux [ppt]")
        ax.set_xlabel("Phase")
        #	ax.set_xlim(0.4, 0.6)
        
        txt = "Planet b Period = {:.3f}".format(map_soln0['P_b'])
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
        
        # phase fold for planet c
        ax = axes[1]
        ax.scatter(phases_c[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
        mod_c = soln["light_curves_c"]
        ax.plot(phases_c[mask][arg_c], mod_c[arg_c], color='blue', label="Planet c Model")
        ax.legend(fontsize=12)
        ax.set_ylabel("De-trended Flux [ppt]")
        #	ax.set_xlim(0.4, 0.6)
        
        txt = "Planet c Period = {:.3f}".format(map_soln0['P_c'])
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
	
    return fig

def plot_light_curve_auto(soln, mask=None,instrument = 'both', planet_num=1):
    if instrument == 'both':
        if mask is None:
        		mask = np.ones(len(time_TESS), dtype=bool)
        if planet_num == 1:
            fig = plt.figure()
#        if planet_num == 6:
#            fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=False)
        else:
            fig, axes = plt.subplots(planet_num, 1, figsize=(8, 10), sharex=False)
        
        plt.title('{}'.format(target_ID))
    	
        # setting up the phase fold data
        # n.b. these need changing so that they use the fully detrended versions, e.g. flux_TESS{something:end}
        phases_b = np.mod(time_TESS - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
        phases_b_TESS = np.mod(time_TESS_orig - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
        phases_b_Kepler = np.mod(time_Kepler - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
        arg_b = np.argsort(phases_b)
        gp_mod = soln["gp_pred"] + soln["mean"]
    	
        # phase fold for planet b
        if planet_num == 1:
            ax = plt.gca()
        else:
            ax = axes[0]
        plt.title('{}'.format(target_ID))
        ax.scatter(phases_b_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1, label="TESS De-trended Data")
        ax.scatter(phases_b_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1, label="Kepler De-trended Data")
        mod_b = soln["light_curves_b"]
#        ax.plot(phases_b[mask][arg_b], mod_b[arg_b]+0.005, color='orange', label="Planet b Model")
        ax.plot(phases_b[mask][arg_b], mod_b[arg_b], color='orange', label="Planet b Model")
        ax.legend(fontsize=12)
        ax.set_ylabel("De-trended Flux [ppt]")
        ax.set_xlabel("Phase")
        ax.set_xlim(0, 1)
        txt = "Planet b Period = {:.3f}".format(map_soln0['P_b'])
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
        
        if planet_num > 1:
            # phase fold for planet c
            phases_c_TESS = np.mod(time_TESS_orig - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
            phases_c_Kepler = np.mod(time_Kepler - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
            phases_c = np.mod(time_TESS - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
            arg_c = np.argsort(phases_c)
            
            ax = axes[1]
            ax.scatter(phases_c_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1)
            ax.scatter(phases_c_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1)
            mod_c = soln["light_curves_c"]
            ax.plot(phases_c[mask][arg_c], mod_c[arg_c], color='blue', label="Planet c Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet c Period = {:.3f}".format(map_soln0['P_c'])
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
        
        if planet_num > 2:
            # phase fold for planet d
            phases_d_TESS = np.mod(time_TESS_orig - soln['t0_d']-soln['P_d']/2, soln['P_d'])/soln['P_d']
            phases_d_Kepler = np.mod(time_Kepler - soln['t0_d']-soln['P_d']/2, soln['P_d'])/soln['P_d']
            phases_d = np.mod(time_TESS - soln['t0_d']-soln['P_d']/2, soln['P_d'])/soln['P_d']
            arg_d = np.argsort(phases_d)
            
            ax = axes[2]
            ax.scatter(phases_d_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1)
            ax.scatter(phases_d_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1)
            mod_d = soln["light_curves_d"]
            ax.plot(phases_d[mask][arg_d], mod_d[arg_d], color='cyan', label="Planet d Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet d Period = {:.3f}".format(map_soln0['P_d'])
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
            
        if planet_num > 3:
            # phase fold for planet e
            phases_e_TESS = np.mod(time_TESS_orig - soln['t0_e']-soln['P_e']/2, soln['P_e'])/soln['P_e']
            phases_e_Kepler = np.mod(time_Kepler - soln['t0_e']-soln['P_e']/2, soln['P_e'])/soln['P_e']
            phases_e = np.mod(time_TESS - soln['t0_e']-soln['P_e']/2, soln['P_e'])/soln['P_e']
            arg_e = np.argsort(phases_e)
            
            ax = axes[3]
            ax.scatter(phases_e_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1)
            ax.scatter(phases_e_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1)
            mod_e = soln["light_curves_e"]
            ax.plot(phases_e[mask][arg_e], mod_e[arg_e], color='g', label="Planet e Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet e Period = {:.3f}".format(map_soln0['P_e'])
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
        
        if planet_num > 4:
            # phase fold for planet f
            phases_f_TESS = np.mod(time_TESS_orig - soln['t0_f']-soln['P_f']/2, soln['P_f'])/soln['P_f']
            phases_f_Kepler = np.mod(time_Kepler - soln['t0_f']-soln['P_f']/2, soln['P_f'])/soln['P_f']
            phases_f = np.mod(time_TESS - soln['t0_f']-soln['P_f']/2, soln['P_f'])/soln['P_f']
            arg_f = np.argsort(phases_f)
            
            ax = axes[4]
            ax.scatter(phases_f_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1)
            ax.scatter(phases_f_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1)
            mod_f = soln["light_curves_f"]
            ax.plot(phases_f[mask][arg_f], mod_f[arg_f], color='red', label="Planet f Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet f Period = {:.3f}".format(map_soln0['P_f'])
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
        
        if planet_num > 5:
            # phase fold for planet g
            phases_g_TESS = np.mod(time_TESS_orig - soln['t0_g']-soln['P_g']/2, soln['P_g'])/soln['P_g']
            phases_g_Kepler = np.mod(time_Kepler - soln['t0_g']-soln['P_g']/2, soln['P_g'])/soln['P_g']
            phases_g = np.mod(time_TESS - soln['t0_g']-soln['P_g']/2, soln['P_g'])/soln['P_g']
            arg_g = np.argsort(phases_g)
            
            ax = axes[5]
            ax.scatter(phases_g_TESS, flux_TESS[len(time_Kepler):]-gp_mod[len(time_Kepler):], c='k', s=1)
            ax.scatter(phases_g_Kepler, flux_TESS[0:len(time_Kepler):]-gp_mod[0:len(time_Kepler):], c='darkgrey', s=1)
            mod_g = soln["light_curves_g"]
            ax.plot(phases_g[mask][arg_g], mod_g[arg_g], color='purple', label="Planet g Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet g Period = {:.3f}".format(map_soln0['P_g'])
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
        
    else:
        if mask is None:
        		mask = np.ones(len(time_TESS), dtype=bool)
        if planet_num == 1:
            fig = plt.figure()
        else:
            fig, axes = plt.subplots(planet_num, 1, figsize=(8, 10), sharex=False)
        
        plt.title('{}'.format(target_ID))

        # setting up the phase fold data
        phases_b = np.mod(time_TESS - soln['t0_b']-soln['P_b']/2, soln['P_b'])/soln['P_b']
        arg_b = np.argsort(phases_b)
        gp_mod = soln["gp_pred"] + soln["mean"]
    	
        # phase fold for planet b
        if planet_num == 1:
            ax = plt.gca()
        else:
            ax = axes[0]
        ax.scatter(phases_b[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
        binned_time_b, binned_flux_b = binned(phases_b[mask][arg_b], flux_TESS[mask][arg_b]-gp_mod[arg_b])
        ax.scatter(binned_time_b, binned_flux_b, c='r', s=1, label="Binned {} De-trended Data".format(instrument))
        mod_b = soln["light_curves_b"]
        ax.plot(phases_b[mask][arg_b], mod_b[arg_b], color='orange', label="Planet b Model")
        ax.legend(fontsize=12)
        ax.set_ylabel("De-trended Flux [ppt]")
        ax.set_xlabel("Phase")
        ax.set_xlim(0, 1)
        txt = "Planet b Period = {:.3f}".format(map_soln0['P_b'])
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
        if planet_num > 1:
            # phase fold for planet c
            phases_c = np.mod(time_TESS - soln['t0_c']-soln['P_c']/2, soln['P_c'])/soln['P_c']
            arg_c = np.argsort(phases_c)
            ax = axes[1]
            ax.scatter(phases_c[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
            binned_time_c, binned_flux_c = binned(phases_c[mask][arg_c], flux_TESS[mask][arg_c]-gp_mod[arg_c])
            ax.scatter(binned_time_c, binned_flux_c, c='r', s=1, label="Binned {} De-trended Data".format(instrument))
            mod_c = soln["light_curves_c"]
            ax.plot(phases_c[mask][arg_c], mod_c[arg_c], color='blue', label="Planet c Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet c Period = {:.3f}".format(map_soln0['P_c'])
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
            
        if planet_num > 2:
            # phase fold for planet d
            phases_d = np.mod(time_TESS - soln['t0_d']-soln['P_d']/2, soln['P_d'])/soln['P_d']
            arg_d = np.argsort(phases_d)
            ax = axes[2]
            ax.scatter(phases_d[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
            binned_time_d, binned_flux_d = binned(phases_d[mask][arg_d], flux_TESS[mask][arg_d]-gp_mod[arg_d])
            ax.scatter(binned_time_d, binned_flux_d, c='r', s=1, label="Binned {} De-trended Data".format(instrument))
            mod_d = soln["light_curves_d"]
            ax.plot(phases_d[mask][arg_d], mod_d[arg_d], color='blue', label="Planet d Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet d Period = {:.3f}".format(map_soln0['P_d'])
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
            
        if planet_num > 3:
            # phase fold for planet e
            phases_e = np.mod(time_TESS - soln['t0_e']-soln['P_e']/2, soln['P_e'])/soln['P_e']
            arg_e = np.argsort(phases_e)
            ax = axes[3]
            ax.scatter(phases_e[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
            binned_time_e, binned_flux_e = binned(phases_e[mask][arg_e], flux_TESS[mask][arg_e]-gp_mod[arg_e])
            ax.scatter(binned_time_e, binned_flux_e, c='r', s=1, label="Binned {} De-trended Data".format(instrument))
            mod_e = soln["light_curves_e"]
            ax.plot(phases_e[mask][arg_e], mod_e[arg_e], color='blue', label="Planet e Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet e Period = {:.3f}".format(map_soln0['P_e'])
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
            
        if planet_num > 4:
            # phase fold for planet f
            phases_f = np.mod(time_TESS - soln['t0_f']-soln['P_f']/2, soln['P_f'])/soln['P_f']
            arg_f = np.argsort(phases_f)
            ax = axes[4]
            ax.scatter(phases_f[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
            binned_time_f, binned_flux_f = binned(phases_f[mask][arg_f], flux_TESS[mask][arg_f]-gp_mod[arg_f])
            ax.scatter(binned_time_f, binned_flux_f, c='r', s=1, label="Binned {} De-trended Data".format(instrument))
            mod_f = soln["light_curves_f"]
            ax.plot(phases_f[mask][arg_f], mod_f[arg_f], color='blue', label="Planet f Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet f Period = {:.3f}".format(map_soln0['P_f'])
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
            
        if planet_num > 5:
            # phase fold for planet f
            phases_g = np.mod(time_TESS - soln['t0_g']-soln['P_g']/2, soln['P_g'])/soln['P_g']
            arg_g = np.argsort(phases_g)
            ax = axes[5]
            ax.scatter(phases_g[mask], flux_TESS[mask]-gp_mod, c='k', s=1, label="{} De-trended Data".format(instrument))
            binned_time_g, binned_flux_g = binned(phases_g[mask][arg_g], flux_TESS[mask][arg_g]-gp_mod[arg_g])
            ax.scatter(binned_time_g, binned_flux_g, c='r', s=1, label="Binned {} De-trended Data".format(instrument))
            mod_g = soln["light_curves_g"]
            ax.plot(phases_g[mask][arg_g], mod_f[arg_g], color='blue', label="Planet g Model")
            ax.legend(fontsize=12)
            ax.set_ylabel("De-trended Flux [ppt]")
            ax.set_xlim(0, 1)
            txt = "Planet g Period = {:.3f}".format(map_soln0['P_g'])
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
	
    return fig

if method == 'manual':
    plot_light_curve(map_soln0, instrument = instrument);
else:
    plot_light_curve_auto(map_soln0, instrument = instrument, planet_num = planet_num);


############## Appendix: Tutorial - A1: Transit Model in PyMC3 ################
#np.random.seed(123)
#periods = np.random.uniform(5, 20, 2)
#t0s = periods * np.random.rand(2)
#t = np.arange(0, 80, 0.02)
#yerr = 5e-4
#
#with pm.Model() as model:
#
#    # The baseline flux
#    mean = pm.Normal("mean", mu=0.0, sd=5.0)
#
#    # The time of a reference transit for each planet
#    t0 = pm.Normal("t0", mu=t0s, sd=1.0, shape=2)
#
#    # The log period; also tracking the period itself
#    logP = pm.Normal("logP", mu=np.log(periods), sd=0.1, shape=2)
#    period = pm.Deterministic("period", pm.math.exp(logP))
#
#    # The Kipping (2013) parameterization for quadratic limb darkening paramters
#    u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
#
#    r = pm.Uniform(
#        "r", lower=0.01, upper=0.1, shape=2, testval=np.array([0.04, 0.06])
#    )
#    b = xo.distributions.ImpactParameter(
#        "b", ror=r, shape=2, testval=np.random.rand(2)
#    )
#
#    # Set up a Keplerian orbit for the planets
#    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
#
#    # Compute the model light curve using starry
#    light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
#        orbit=orbit, r=r, t=t
#    )
#    light_curve = pm.math.sum(light_curves, axis=-1) + mean
#
#    # Here we track the value of the model light curve for plotting
#    # purposes
#    pm.Deterministic("light_curves", light_curves)
#
#    # In this line, we simulate the dataset that we will fit
##    y = xo.eval_in_model(light_curve)
##    y += yerr * np.random.randn(len(y))
#
#    # The likelihood function assuming known Gaussian uncertainty
#    pm.Normal("obs", mu=light_curve, sd=yerr, observed=flux)
#
#    # Fit for the maximum a posteriori parameters given the simuated
#    # dataset
#    map_soln = xo.optimize(start=model.test_point)
#
#plt.plot(t, flux, ".k", ms=4, label="data")
#for i, l in enumerate("bc"):
#    plt.plot(
#        t, map_soln["light_curves"][:, i], lw=1, label="planet {0}".format(l)
#    )
#plt.xlim(t.min(), t.max())
#plt.ylabel("relative flux")
#plt.xlabel("time [days]")
#plt.legend(fontsize=10)
#_ = plt.title("map model")

################################ A2: Sampling #################################

#np.random.seed(42)
#with model:
#    trace = pm.sample(
#        tune=3000,
#        draws=3000,
#        start=map_soln0,
#        cores=2,
#        chains=2,
#        step=xo.get_dense_nuts_step(target_accept=0.9),
#    )
#    
#pm.summary(trace, varnames=["period", "t0", "r", "b", "u", "mean"])
#
#samples = pm.trace_to_dataframe(trace, varnames=["period", "r"])
#truth = np.concatenate(
#    xo.eval_in_model([period, r], model.test_point, model=model)
#)
#_ = corner.corner(
#    samples,
#    truths=truth,
#    labels=["period 1", "period 2", "radius 1", "radius 2"],
#)
#
############################### A3: Phase Plots ###############################
#
#for n, letter in enumerate("bc"):
#    plt.figure()
#
#    # Get the posterior median orbital parameters
#    p = np.median(trace["period"][:, n])
#    t0 = np.median(trace["t0"][:, n])
#
#    # Compute the median of posterior estimate of the contribution from
#    # the other planet. Then we can remove this from the data to plot
#    # just the planet we care about.
#    other = np.median(trace["light_curves"][:, :, (n + 1) % 2], axis=0)
#
#    # Plot the folded data
#    x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
#    plt.errorbar(
#        x_fold, flux - other, yerr=yerr, fmt=".k", label="data", zorder=-1000
#    )
#
#    # Plot the folded model
#    inds = np.argsort(x_fold)
#    inds = inds[np.abs(x_fold)[inds] < 0.3]
#    pred = trace["light_curves"][:, inds, n] + trace["mean"][:, None]
#    pred = np.median(pred, axis=0)
#    plt.plot(x_fold[inds], pred, color="C1", label="model")
#
#    # Annotate the plot with the planet's period
#    txt = "period = {0:.4f} +/- {1:.4f} d".format(
#        np.mean(trace["period"][:, n]), np.std(trace["period"][:, n])
#    )
#    plt.annotate(
#        txt,
#        (0, 0),
#        xycoords="axes fraction",
#        xytext=(5, 5),
#        textcoords="offset points",
#        ha="left",
#        va="bottom",
#        fontsize=12,
#    )
#
#    plt.legend(fontsize=10, loc=4)
#    plt.xlim(-0.5 * p, 0.5 * p)
#    plt.xlabel("time since transit [days]")
#    plt.ylabel("relative flux")
#    plt.title("planet {0}".format(letter))
#    plt.xlim(-0.3, 0.3)
