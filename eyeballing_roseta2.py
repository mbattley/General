#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:23:20 2020

Eyeballing_roseta2.py

Script to speed up the eyeballing process for a set of previously detrended light-curves.

@author: mbattley
"""
import csv
import lightkurve
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from lc_download_methods import get_lc_from_fits, lc_from_csv
plt.ion()

def get_lc_from_flat_csv(filename):
    with open(filename, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        time = next(csv_reader)
        flux = next(csv_reader)
        time = [float(i) for i in time]
        flux = [float(i) for i in flux]
        lc = lightkurve.LightCurve(time = time, flux = flux)
    return lc

#save_path = "/Home/Roseta/"
save_path = '/Users/mbattley/Documents/PhD/Roseta/'

#data = Table.read(save_path + 'Period_info_table_S14_per_geq1.csv',format='ascii.csv')
data = Table.read(save_path + 'Period_info_table_new.csv')
full_TIC_list = data['TIC']
full_period_list = data['Max Period']
full_epoch_list = data['Epoch of Max']
#print(TIC_list)

#TIC_list = [159216376,11695827,188519103,359675849,185541586,102604552,47757779,354575514,339622959,91701780]
TIC_list = [11695746,188518505,138717209,59378294]

with open(save_path + 'Eyeballing_notes_test.csv','w') as f:
    header_row = ['TIC','Eyeballing Notes']
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_row)

for tic in TIC_list:
    detrended_filename = save_path + '{}_detrended_lc.csv'.format(tic)
    lowess_filename = save_path + '{}_lowess_fit_lc.csv'.format(tic)
    periodogram_filename = save_path + '{}_periodogram_data.csv'.format(tic)
    
    #Open original lc
    target_ID_pad = str(tic).rjust(16,'0')
    filename = 'hlsp_qlp_tess_ffi_s0014-{}{}{}{}_tess_v01_llc.fits'.format(target_ID_pad[0:4],target_ID_pad[4:8],target_ID_pad[8:12],target_ID_pad[12:16])
    og_lc = get_lc_from_fits(filename, source = 'QLP', clean=True)
    
    #Open results from detrending and bls search
    detrended_lc = lc_from_csv(detrended_filename)
    lowess_lc = lc_from_csv(lowess_filename)
    periodogram_lc = get_lc_from_flat_csv(periodogram_filename)
    
    #Get bls period/epoch info for object
    i = list(full_TIC_list).index(tic)
    period = full_period_list[i]
    epoch = float(full_epoch_list[i][:-1])
    
    #Plot overview plot
    eye_balling_fig, axs = plt.subplots(2,2, figsize = (16,10),  dpi = 120)

    # Original DIA with injected transits setup
    axs[0,0].scatter(og_lc.time, og_lc.flux, s=1, c= 'k')
    axs[0,0].plot(lowess_lc.time,lowess_lc.flux)
    axs[0,0].set_ylabel('Normalized Flux')
    axs[0,0].set_xlabel('Time- 2457000 [BTJD days]')
    axs[0,0].set_title('{} light curve'.format(tic))
    
    
    # Detrended figure setup
    axs[0,1].scatter(detrended_lc.time, detrended_lc.flux, c = 'k', s = 1)
#    axs[0,1].set_title('{} residuals after {} detrending - Sector {}'.format(target_ID, detrending, sector))
    axs[0,1].set_ylabel('Normalized Flux')
    axs[0,1].set_xlabel('Time - 2457000 [BTJD days]')

    
    # Periodogram setup
    axs[1,0].plot(periodogram_lc.time, periodogram_lc.flux, "k", lw=0.5)
    axs[1,0].set_xlim(min(periodogram_lc.time), max(periodogram_lc.time))
    axs[1,0].set_xlabel("period [days]")
    axs[1,0].set_ylabel("log likelihood")
#    axs[1,0].set_title('{} - BLS Periodogram of residuals'.format(target_ID))
    axs[1,0].axvline(period, alpha=0.4, lw=3)
    for n in range(2, 10):
        axs[1,0].axvline(n*period, alpha=0.4, lw=1, linestyle="dashed")
        axs[1,0].axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")
    
    # Folded or zoomed plot setup
    print('Main epoch is {}'.format(epoch))
    phase = np.mod(detrended_lc.time-epoch-period/2,period)/period 
    axs[1,1].scatter(phase,detrended_lc.flux, c='k', s=1)
    axs[1,1].set_title('{} Lightcurve folded by {:0.4} days'.format(tic, period))
    axs[1,1].set_xlabel('Phase')
    axs[1,1].set_ylabel('Normalized Flux')
    #axs[1,1].set_xlim(0.4,0.6)
#    binned_phase, binned_lc = bin(phase, BLS_flux, binsize=15, method='mean')
#    plt.scatter(binned_phase, binned_lc, c='r', s=4)

#    eye_balling_fig.tight_layout()
#            plt.close(eye_balling_fig)
#    plt.ion()
    plt.show(block=False)
    plt.pause(10)
#    plt.close(eye_balling_fig)
    
    print(tic)
    eyeballing_notes = input('Enter eyeballing notes:')
    
    with open(save_path + 'Eyeballing_notes_test.csv','a') as f:
        data_row = [str(tic),eyeballing_notes]
        writer = csv.writer(f, delimiter=',')
        writer.writerow(data_row)
    
    #plt.close(eye_balling_fig)
    