#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:44:34 2020

Kepler/TESS Planet Modelling
Allows a quick check of whether Kepler planets around stars reobserved by TESS
fall into observation window, and if so simultaneously models TESS and Kepler 
data for planet

@author: mbattley
"""
import batman
import time
import csv
import lightkurve
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from lc_download_methods import two_min_lc_download
from remove_tess_systematics import clean_tess_lc


def get_transit_params(target_id= '',source='from_file', planet_vs_pc = 'planet'):
    return

def my_custom_corrector_func(lc):
    corrected_lc = lc.normalize().flatten(window_length=401)
    return corrected_lc

start = time.time()

plt.rcParams['figure.dpi'] = 300
################################### INPUTS ####################################
save_path = '/Users/mbattley/Documents/PhD/Kepler-2min xmatch/'
target_ID = 27769688 #'272369124' #TIC 272369124/Kepler-34 b
sector = 14
multi_sector = False
source = 'planet'
planet_params = 'user_defined' # n.b. can be 'user_defined' or 'from_file'
colours = ['r', 'b', 'g', 'c', 'm', 'y']
planets = ['b', 'c', 'd', 'e', 'f', 'g']
###############################################################################

#with open(save_path+'Planet epochs/Failed_lcs.csv','w') as f:
#    info_row = ['TIC', 'Sector']
#    writer = csv.writer(f, delimiter=',')
#    writer.writerow(info_row)

planet_data = Table.read(save_path + 'Kepler_planets_reobserved_in_TESS_2min.csv', format='ascii.csv')
#pc_data = Table.read(save_path + 'Kepler_pcs_reobserved_in_TESS_2min_final.csv', format='ascii.csv')

num_lcs = 0
#target_ID_list = np.array(pc_data['TICID'])
target_ID_list = np.array([27769688])
for target_ID in target_ID_list:
    try:
        multi_sector = False
        
        i = list(planet_data['TICID']).index(int(target_ID))
        print('Got here')
        if source == 'planet':
            per = planet_data['pl_orbper'][i]
            t0_TESS = planet_data['pl_tranmid'][i]   - 2457000
            t0_Kepler = planet_data['pl_tranmid'][i] - 2454833
        elif source == 'pc':
            per = pc_data['koi_period'][i]
            t0_TESS = pc_data['koi_time0bk'][i] - (2457000 - 2454833)
            t0_Kepler = pc_data['koi_time0bk'][i] 
        #target_ID = planet_data['TICID'][71]
        
        if (planet_data['S14'][i] != 0) and (planet_data['S15'][i] != 0):
            multi_sector = [14,15]
        elif (planet_data['S14'][i] != 0) and (planet_data['S26'][i] != 0):
            multi_sector = [14,26]
        
        ##################################### TESS lc #################################
        # Obtain TESS lc
        lc = lightkurve.lightcurve.TessLightCurve(time = [],flux=[])
        if multi_sector != False:
            sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = multi_sector[0], from_file = False)
            lc = pdcsap_lc
            nancut = np.isnan(lc.flux) | np.isnan(lc.time)
            lc = lc[~nancut]
            clean_time, clean_flux, clean_flux_err = clean_tess_lc(lc.time, lc.flux, lc.flux_err, target_ID, multi_sector[0], save_path)
            lc.time = clean_time
            lc.flux = clean_flux
            lc.flux_err = clean_flux_err
            for sector_num in multi_sector[1:]:
                sap_lc_new, pdcsap_lc_new = two_min_lc_download(target_ID, sector_num, from_file = False)
                lc_new = pdcsap_lc_new
                nancut = np.isnan(lc_new.flux) | np.isnan(lc_new.time)
                lc_new = lc_new[~nancut]
        #        clean_time, clean_flux, clean_flux_err = clean_tess_lc(lc_new.time, lc_new.flux, lc_new.flux_err, target_ID, sector_num, save_path)
        #        lc_new.time = clean_time
        #        lc_new.flux = clean_flux
        #        lc_new.flux_err = clean_flux_err
                lc = lc.append(lc_new)
        else:
            sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = sector, from_file = False)
            lc = pdcsap_lc
            nancut = np.isnan(lc.flux) | np.isnan(lc.time)
            lc = lc[~nancut]
            print('Removed nans')
            
        #lc.scatter()
        
        tess_lc = lc.normalize().flatten(window_length=401)
        tess_fig = tess_lc.scatter()
        #tess_fig = plt.figure()
        #plt.scatter(tess_lc.time,tess_lc.flux, c='k', s=2)
        #plt.xlabel('Time - 2457000 [BTJD days]')
        #plt.ylabel("Normalized Flux")
        
        # Add lines to TESS plot
        ax = plt.gca()
        planet = 0
        if source == 'planet':
            for j in range(len(planet_data['TICID'])):
                print(planet_data['TICID'][j])
                if planet_data['TICID'][j] == target_ID:
                    per = planet_data['pl_orbper'][j]
                    t0_TESS = planet_data['pl_tranmid'][j] - 2457000
                    line_time = t0_TESS
                    while line_time < tess_lc.time[0]:
                        line_time += per
                    if line_time > tess_lc.time[-1]:
                        ax.annotate('Epoch for planet {} falls outside TESS window'.format(planets[planet]), xy=(2, 1))
                    else:
                        while line_time < tess_lc.time[-1]:
                            ax.axvline(line_time, ymin = 0.1, ymax = 0.2, lw=1, c=colours[planet])
                            line_time += per
                    planet += 1
                    print(per, t0_TESS)
        if source == 'pc':
            for j in range(len(pc_data['TICID'])):
                if pc_data['TICID'][j] == target_ID:
                    per = pc_data['koi_period'][j]
                    t0_TESS = pc_data['koi_time0bk'][j] - (2457000 - 2454833)
                    line_time = t0_TESS
                    while line_time < tess_lc.time[0]:
                        line_time += per
                    if line_time > tess_lc.time[-1]:
                        ax.annotate('Epoch for candidate {} falls outside TESS window'.format(planets[planet]), xy=(2, 1))
                    else:
                        while line_time < tess_lc.time[-1]:
                            ax.axvline(line_time, ymin = 0.1, ymax = 0.2, lw=1, c=colours[planet])
                            line_time += per
                    planet += 1
        
#        tess_fig.figure.savefig(save_path + "Planet epochs/" + "TIC {} - TESS candidate epochs.png".format(target_ID))
#        plt.close(tess_fig.figure)
        
        ############################### KEPLER lc #####################################
        
#        # Obtain all Kepler lcs
#        lcfs = lightkurve.search_lightcurvefile(planet_data['kepid'][i], mission='Kepler').download_all()
#        stitched_lc = lcfs.PDCSAP_FLUX.stitch()
#        #stitched_lc.scatter()
#        stitched_lc = lcfs.PDCSAP_FLUX.stitch(corrector_func=my_custom_corrector_func)
#        kepler_fig = stitched_lc.scatter()
#        #kepler_fig = plt.figure()
#        #plt.scatter(stitched_lc.time,stitched_lc.flux,c='k', s=2)
#        #plt.xlabel('Time - 2454833 [BKJD days]')
#        #plt.ylabel("Normalized Flux")
#         
#        # Add lines to Kepler plot
#        ax = plt.gca()
#        line_time = t0_Kepler
#        #while line_time < stitched_lc.time[-1]:
#        #    ax.axvline(line_time, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#        #    line_time += per
#        planet = 0
#        if source == 'planet':
#            for j in range(len(planet_data['TICID'])):
#                if planet_data['TICID'][j] == target_ID:
#                    per = planet_data['pl_orbper'][j]
#                    t0_Kepler = planet_data['pl_tranmid'][j] - 2454833
#                    line_time = t0_Kepler
#                    while line_time < stitched_lc.time[-1]:
#                        ax.axvline(line_time, ymin = 0.1, ymax = 0.2, lw=1, c=colours[planet])
#                        line_time += per
#                    planet += 1
#        elif source == 'pc':
#            for j in range(len(pc_data['TICID'])):
#                if pc_data['TICID'][j] == target_ID:
#                    per = pc_data['koi_period'][j]
#                    t0_Kepler = pc_data['koi_time0bk'][j]
#                    line_time = t0_Kepler
#                    while line_time < stitched_lc.time[-1]:
#                        ax.axvline(line_time, ymin = 0.1, ymax = 0.2, lw=1, c=colours[planet])
#                        line_time += per
#                    planet += 1
        
#        kepler_fig.figure.savefig(save_path + "Planet epochs/"+ "TIC {} - Kepler planet epochs.png".format(target_ID))
#        plt.close(kepler_fig.figure)
    except:
        print('Planet failed you fool')
#        with open(save_path+'Candidate epochs/Failed_lcs.csv','a') as f:
#            data_row = [target_ID, sector]
#            writer = csv.writer(f, delimiter=',')
#            writer.writerow(data_row)
    num_lcs += 1
    print("Number of light-curves analysed: {}".format(num_lcs))
    partway = time.time()
    print("Elapsed time partway: {}".format(partway-start))

end = time.time()
print("Final elapsed time: {}".format(end-start))


#params_array = np.array([])
## Get planet parameters for each planet
#for i in range(len(planet_data['TICID'])):
#    if planet_data['TICID'][i] == target_ID:
#        params = batman.TransitParams()       #object to store transit parameters
#        if planet_params == 'user_defined':
#            params.t0 = 2454969.2 - 2457000   #time of inferior conjunction - converted to TJD
#            params.per = 88.82
#            params.rp = 0.1
#            #For a: 25 for 10d; 17 for 8d; 10 for 4d; 4-8 (6) for 2 day; 2-5  for 1d; 1-3 (or 8?) for 0.5d
#            params.a = 17. #semi-major axis (in units of stellar radii)
#            params.inc = 90.
#            params.ecc = 0.
#            params.w = 90.                        #longitude of periastron (in degrees)
#            params.limb_dark = "nonlinear"        #limb darkening model
#            params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]
#        elif planet_params == 'from_file' and source == 'planet':
#            params.t0 = 2454969.2 - 2457000   #time of inferior conjunction - converted to TJD
#            params.per = 88.82
#            params.rp = 0.1
#            params.a = 17. #semi-major axis (in units of stellar radii)
#            params.inc = 90.
#            params.ecc = 0.
#            params.w = 90.                        #longitude of periastron (in degrees)
#            params.limb_dark = "nonlinear"        #limb darkening model
#            params.u = [0.5, 0.1, 0.1, -0.1]
#        print(params)
#        np.append(params_array,params)
#    
#for params in params_array:
#    # Defines times at which to calculate lc and models batman lc
#    t = lc.time
#    index = int(len(lc.time)//2)
#    mid_point = lc.time[index]
#    t = lc.time - lc.time[index]
#    m = batman.TransitModel(params, t)
#    t += lc.time[index]
#    #        print("About to compute flux")
#    batman_flux = m.light_curve(params)
#    #        print("Computed flux")
#    batman_model_fig = plt.figure()
#    plt.scatter(lc.time, batman_flux, s = 2, c = 'k')
#    plt.xlabel("Time - 2457000 (BTJD days)")
#    plt.ylabel("Relative flux")
#    plt.title("batman model transit for {}R ratio".format(params.rp))
##    batman_model_fig.savefig(save_path + "batman model transit for {}d {}R planet.png".format(params.per,params.rp))
##    plt.close(batman_model_fig)
#    plt.show()
