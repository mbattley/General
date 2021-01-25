#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:08:51 2021

merge_lcs.py

Short code to merge a series of lightcurves into a single file

@author: mbattley
"""

import pickle
import scipy
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy import optimize
from astropy.timeseries import LombScargle
from lc_download_methods import diff_image_lc_download
from remove_tess_systematics import clean_tess_lc

#def find_freqs(time, flux, plot_ls_fig = True):
#    
#     #From Lomb-Scargle
#    freq = np.arange(0.05,4.1,0.00001)
#    power = LombScargle(time, flux).power(freq)
#    if plot_ls_fig == True:
#        ls_fig = plt.figure()
#        plt.plot(freq, power, c='k', linewidth = 1)
#        plt.xlabel('Frequency')
#        plt.ylabel('Power')
#        plt.title('{} LombScargle Periodogram for original lc'.format(target_ID))
#        ls_fig.show()
##        ls_fig.savefig(save_path + '{} - Lomb-Sacrgle Periodogram for original lc.png'.format(target_ID))
##        plt.close(ls_fig)
#    i = np.argmax(power)
#    freq_rot = freq[i]
#    
#    # Find indices of 2nd and 3rd peaks of periodogram
#    all_peaks = scipy.signal.find_peaks(power, width = 5, distance = 10)[0]
#    all_peak_powers = power[all_peaks]
#    sorted_power_indices = np.argsort(all_peak_powers)
#    sorted_peak_powers = all_peak_powers[sorted_power_indices]
#    sorted_peak_freqs =freq[all_peaks][sorted_power_indices]
#    
#    # MASKS ALL HARMONICS OF MAXIMUM POWER PERIOD
#    # n.b. Uses a similar technique to masking bad times is tess_systematics cleaner:
#    # E.g. mask all those close to 1.5, 2, 2.5, 3, 3.5, 4, ..., 10 etc times the max period
#    # and then search through rest as normal
##    harmonic_mask = [False]*len(sorted_peak_powers)
##    harmonics = period.value*np.array([0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,4.5,5,5.5,6,
##                                       6.5,7,7.5,8,8.5,9,9.5,10])
##    simplified_harmonics = freq_rot*np.array([1/3,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9,10])
##            print('Created Harmonics')
##    
##    for i in simplified_harmonics:
##        for j in range(len(sorted_peak_freqs)):
###                    print(sorted_peak_periods[j].value - i)
##            if abs(sorted_peak_freqs[j] - i) < 0.01:
##                harmonic_mask[j] = True
##            print('Completed for loop')
##    harmonic_mask = np.array(harmonic_mask)
##    sorted_peak_powers = sorted_peak_powers[~harmonic_mask]
##    sorted_peak_freqs = sorted_peak_freqs[~harmonic_mask]
##    
#    # Find info for 2nd largest peak in periodogram
#    index_peak_2 = np.where(power==sorted_peak_powers[-2])[0]
#    freq_2 = freq[index_peak_2[0]]
#    
#    # Find info for 3rd largest peak in periodogram
#    index_peak_3 = np.where(power==sorted_peak_powers[-3])[0]
#    freq_3 = freq[index_peak_3[0]]
#
#    freq_list = [freq_rot,freq_2,freq_3]
#    
#    return freq_list

def trig_func(t,f,a,b,c):
    return a*np.sin(2*np.pi*f*t) + b*np.cos(2*np.pi*f*t) + c

def next_highest_freq(time, flux, freq, f_remove, plot_ls_fig = False):
    popt_maxV, pcov_maxV = optimize.curve_fit(lambda t, a, b, c: trig_func(t,f_remove, a, b, c), time, flux, maxfev=1000)
    max_var = trig_func(time,f_remove,*popt_maxV)
    flux = flux/max_var
    power = LombScargle(time, flux).power(freq)
    if plot_ls_fig == True:
        ls_fig = plt.figure()
        plt.plot(freq, power, c='k', linewidth = 1)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('{} LombScargle Periodogram'.format(target_ID))
        ls_fig.show()
    i = np.argmax(power)
    freq_2 = freq[i]
    return freq_2, flux

def find_freqs(time, flux, plot_ls_fig = True):
    
    # Remove frequencies associated with 14d data gap
    f=1/14
    popt_sys, pcov_sys = optimize.curve_fit(lambda t, a, b, c: trig_func(t,f, a, b, c), time, flux, maxfev=1000)
    sys_var = trig_func(time,f,*popt_sys)
    flux = flux/sys_var
    
     #From Lomb-Scargle
    freq = np.arange(0.05,4.1,0.00001)
    power = LombScargle(time, flux).power(freq)
    if plot_ls_fig == True:
        ls_fig = plt.figure()
        plt.plot(freq, power, c='k', linewidth = 1)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('{} LombScargle Periodogram for original lc'.format(target_ID))
        ls_fig.show()
#        ls_fig.savefig(save_path + '{} - Lomb-Scargle Periodogram for original lc.png'.format(target_ID))
#        plt.close(ls_fig)
    i = np.argmax(power)
    freq_rot = freq[i]
    
    # Remove highest frequency to get 2nd highest
    f_remove=freq_rot
    freq_2, flux = next_highest_freq(time, flux, freq, f_remove, plot_ls_fig = False)
        
    # Remove 2nd highest frequency to get 3rd highest
    f_remove=freq_2
    freq_3, flux = next_highest_freq(time, flux, freq, f_remove, plot_ls_fig = False)

    freq_list = [freq_rot,freq_2,freq_3]
    
    #final_fig = plt.figure()
    #plt.scatter(time,flux,s=1,c='k')
    #plt.show()
    
    return freq_list

with open('Target_Lists/Sector_1_targets_from_TIC_list.pkl', 'rb') as f:
    sector_targets = pickle.load(f)
target_ID_list = [str(i) for i in sector_targets]


#target_ID_list = ['J0156-7457','2MASS J01231125-6921379'] #Sector 1 test
#target_ID_list = ['HD 35289','HD 33512'] #Sector 5 test

sector = 5
lc_num = 0
num_done = 0
clean_lcs = True
flux_table = np.ones([510,1341])
freq_table = np.ones([510,3])
mean_errs = np.ones(510)
used_targets = []
sector_list = [1,2,3,4,5]

timearray = np.linspace(0,27.916667,1341)

for sector in sector_list:
    with open('Target_Lists/Sector_{}_targets_from_TIC_list.pkl'.format(sector), 'rb') as f:
        sector_targets = pickle.load(f)
    target_ID_list = [str(i) for i in sector_targets]
    for target_ID in target_ID_list:
        try:
            lc, filename = diff_image_lc_download(target_ID, sector, plot_lc=False)
            
            if clean_lcs == True:
                clean_time, clean_flux, clean_flux_err = clean_tess_lc(lc.time, lc.flux, lc.flux_err, target_ID, sector, save_path='')
                time = clean_time
                flux = clean_flux
                err = clean_flux_err
            else:
                time = lc.time
                flux = lc.flux
                err = lc.flux_err
            freq_list = find_freqs(time,flux,plot_ls_fig = False)
            freq_table[lc_num] = freq_list
            mean_errs[lc_num] = np.mean(err)
            
            time = time - time[0]
            mapped_flux = list(np.zeros(1341) + np.nan)
            for j, item_og in enumerate(time):
                for i, item in enumerate(timearray):
                    if abs(item - item_og) < 0.001:
                        mapped_flux[i] = flux[j]
        #    flux = list(np.append(flux, np.zeros(1341-len(flux)) + np.nan))
            print('Array length = {}'.format(len(mapped_flux)))
            flux_table[lc_num] = mapped_flux
            lc_num += 1
            used_targets.append(target_ID)
        except:
            print('Failed for {}'.format(target_ID))
        print('{} light-curves analysed'.format(num_done))
        num_done +=1
    
np.savetxt('young_star_merged_lcs_S1-5.txt',flux_table)
np.savetxt('young_star_freqs_S1-5.txt',freq_table)
np.savetxt('used_targets_S1-5.txt',used_targets,fmt="%s")
np.savetxt('young_star_mean_errs_S1-5.txt',mean_errs)

#dat = np.genfromtxt('young_star_merged_lcs.txt')
#dat_2 = np.genfromtxt('young_star_freqs.txt')