#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:25:31 2019
Collects different functions for downloading and preparing lightcurves via 
different pathways
Inputs:
    - target_ID for object
    - sector of interest
    - [OPTIONAL] csv file containing target list and their respective TIC number
n.b. all lightcurves returned in lightkurvefile-like format 
i.e. time = lc.time, flux = lc.flux, flux_err = lc.flux_err
@author: Matthew Battley
"""

import lightkurve
import eleanor
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
from lightkurve import search_lightcurvefile
from numpy import loadtxt
from astroquery.mast import Tesscut
from astroquery.mast import Catalogs
from astropy.io import fits

def two_min_lc_download(target_ID, sector, from_file = True):
    """
    Downloads and returns SAP and PDCSAP 2-min lightcurves from MAST
    """
    if from_file == True:
        # reads input table for targets
        table_data = Table.read("K2_2min_overlap_observed.csv" , format='ascii.csv')
        
        # Obtains ra and dec for object from target_ID
        i = list(table_data['TICID']).index(target_ID)
        print(i)
        #camera = table_data['S{}'.format(sector)][i]
#        tic = table_data['MatchID'][i]
        tic = 'TIC ' + str(target_ID)
        
        # Find sector
        sector_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        for sector_num in sector_list:
            if table_data['S'+str(sector_num)][i] != 0:
                sector = sector_num
    else:
        tic = 'TIC ' + str(target_ID)
#        tic = target_ID
    print(tic)
    lcf = search_lightcurvefile(tic, sector=sector).download()
#    lcf = search_lightcurvefile(tic).download()
#    sector = 
    
    # Seperate lightcurves
    sap_lc = lcf.SAP_FLUX
    pdcsap_lc = lcf.PDCSAP_FLUX
    
    return sap_lc, pdcsap_lc

def k2_lc_download(target_ID, campaign, from_file = True):
    """
    Downloads and returns SAP and PDCSAP 2-min lightcurves from K2
    """
    lcf = search_lightcurvefile(target_ID, campaign).download
    sap_lc = lcf.SAP_FLUX
    pdcsap_lc = lcf.PDCSAP_FLUX
    
    return sap_lc, pdcsap_lc


def raw_FFI_lc_download(target_ID, sector, plot_tpf = False, plot_lc = False, save_path = '', from_file = False):
    """
    Downloads and returns 30min cadence lightcurves based on SAP analysis of 
    the raw FFIs
    """    
    if from_file == True:
        with open('Sector_1_target_filenames.pkl', 'rb') as f:
            target_filenames = pickle.load(f)
        f.close()
    else:
        target_filenames = {}
    
        # Find ra, dec and tic # via the TIC (typically based on Gaia DR2)
        TIC_table = Catalogs.query_object(target_ID, catalog = "TIC")
        ra = TIC_table['ra'][0]
        dec = TIC_table['dec'][0]
        tic = TIC_table['ID'][0]
         
        object_coord = SkyCoord(ra, dec, unit="deg")
        manifest = Tesscut.download_cutouts(object_coord, [11,11], path = './TESS_Sector_5_cutouts')
#        sector_info = Tesscut.get_sectors(object_coord)
        if len(manifest['Local Path']) == 1:
            target_filenames[target_ID] = manifest['Local Path'][0][2:]
        elif len(manifest['Local Path']) > 1:
            target_filenames[target_ID] = []
            for filename in manifest['Local Path']:
                target_filenames[target_ID].append(filename[2:])
        else:
            print('Cutout for target {} can not be downloaded'.format(target_ID))
        
    if type(target_filenames[target_ID]) == str:
        filename = target_filenames[target_ID]
    else:
        filename = target_filenames[target_ID][0]
        
    
    # Load tpf
    tpf_30min = lightkurve.search.open(filename)
    
    # Attach target name to tpf
    tpf_30min.targetid = target_ID
    
    # Create a median image of the source over time
    median_image = np.nanmedian(tpf_30min.flux, axis=0)
    
    # Select pixels which are brighter than the 85th percentile of the median image
    aperture_mask = median_image > np.nanpercentile(median_image, 85)
    
    # Plot and save tpf
    if plot_tpf == True:
        tpf_30min.plot(aperture_mask = aperture_mask)
    #tpf_plot.savefig(save_path + '{} - Sector {} - tpf plot.png'.format(target_ID, tpf.sector))
    #plt.close(tpf_plot)
    
    # Convert to lightcurve object
    lc_30min = tpf_30min.to_lightcurve(aperture_mask = aperture_mask)
#    lc_30min = lc_30min[(lc_30min.time < 1346) | (lc_30min.time > 1350)]
    if plot_lc == True:
        lc_30min.scatter()
        plt.title('{} - 30min FFI base lc'.format(target_ID))
        plt.xlabel("Time - 2457000 (BTJD days)")
        plt.ylabel("Relative flux")
        plt.show()
    
    return lc_30min

def diff_image_lc_download(target_ID, sector, plot_lc = True, from_file = True, save_path = ''):
    """
    Downloads and returns 30min cadence lightcurves based on Oelkers & Stassun
    difference imaging analysis method of lightcurve extraction
    """
    if sector < 3:
        DIAdir = '/tess/photometry/DIA_FFI/S{}/clean/'.format(sector)
    else:
        DIAdir = '/tess/photometry/DIA_FFI/S{}/lc/clean/'.format(sector)
    
    if from_file == True:
        # reads input table for targets
        table_data = Table.read("BANYAN_XI-III_members_with_TIC.csv" , format='ascii.csv')
        
        # Obtains ra and dec for object from target_ID
        i = list(table_data['main_id']).index(target_ID)
        ra = table_data['ra'][i]
        dec = table_data['dec'][i]
        #camera = table_data['S{}'.format(sector)][i]
        tic = table_data['MatchID'][i]
    else:
         # Find ra, dec and tic # via the TIC (typically based on Gaia DR2)
         TIC_table = Catalogs.query_object(target_ID, catalog = "TIC")
         ra = TIC_table['ra'][0]
         dec = TIC_table['dec'][0]
         tic = TIC_table['ID'][0]
         
    
    object_coord = SkyCoord(ra, dec, unit="deg")
    sector_info = Tesscut.get_sectors(object_coord)
    
    for i in range(len(sector_info)):
        if sector_info[i][1] == sector:
            index = i
            
    camera = sector_info[index][2]
    ccd = sector_info[index][3]
    
#    star = eleanor.Source(coords=(ra, dec), sector=1)
##    camera = 
#    ccd = star.chip
    
    filename = DIAdir+'{}_sector0{}_{}_{}.lc'.format(tic, sector, camera, ccd)
#    filename = '{}_sector0{}_{}_{}.lc'.format(tic, sector, camera, ccd)
#    filename = '410214986_sector01_3_2.lc'

    print('Trying file {}'.format(filename))
    try:
        lines = loadtxt(filename, delimiter = ' ') # For when in local directory
#        lines = loadtxt(DIAdir+filename, delimiter = ' ') # For when on ngtshead
        DIA_lc = list(map(list, zip(*lines)))
        DIA_mag = np.array(DIA_lc[1])
        
        # Convert TESS magnitudes to flux
        DIA_flux =  10**(-0.4*(DIA_mag - 20.60654144))
        norm_flux = DIA_flux/np.median(DIA_flux)
        
        # Plot Difference imaged data
        if plot_lc == True:
            
            diffImage_fig = plt.figure()
            plt.scatter(DIA_lc[0], norm_flux, s=1, c= 'k')
            plt.ylabel('Normalized Flux')
            plt.xlabel('Time')
            plt.title('{} - Difference imaged light curve from FFIs'.format(target_ID))
            #diffImage_fig.savefig(save_path + '{} - Sector {} - DiffImage flux.png'.format(target_ID, sector))
            plt.close(diffImage_fig)
            diffImage_fig.show()
        
        lc = lightkurve.LightCurve(time = DIA_lc[0],flux = norm_flux, flux_err = DIA_lc[2], targetid = target_ID)
        print('Got the light-curve out')
        
        return lc, filename
    except:
        print('The file {} does not exist - difference imaging data not available for {}'.format(filename,target_ID))

def eleanor_lc_download(target_ID, sector, plot_raw = False, plot_corr = False, plot_pca = False, from_file = False, save_path = ''):
    """
    Downloads and returns the various lightcurves produced by the eleanor pipeline:
        raw_lc = lc with flux from simple aperture photometry
        corr_lc = lc with flux 'corrected' for poitning errors etc (though often not trustworthy)
        pca_lc = lc with flux based on principal component analysis
        psf_lc = lc with flux based on point spread function modelling - n.b. sometimes has problems depending on tensorflow version in python
    """
    if from_file == True:
        table_data = Table.read("BANYAN_XI-III_members_with_TIC.csv" , format='ascii.csv')
        
        # Obtains ra and dec for object from target_ID
        i = list(table_data['main_id']).index(target_ID)
        ra = table_data['ra'][i]
        dec = table_data['dec'][i]
        tic = table_data['MatchID'][i]
    else:
        # Find ra, dec and tic # via the TIC (typically based on Gaia DR2)
        TIC_table = Catalogs.query_object(target_ID, catalog = "TIC")
#        ra = TIC_table['ra'][0]
#        dec = TIC_table['dec'][0]
#        tic = TIC_table['ID'][0]
        tic = target_ID
    
    # Locates star in data
    star = eleanor.Source(tic, sector = sector)
    #star = eleanor.Source(coords=(49.4969, -66.9268), sector=1)
    
    # Extract target pixel file, perform aperture photometry and complete some systematics corrections
    data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False)
    
    q = data.quality == 0
    
    # Plot raw flux
    raw_lc = lightkurve.LightCurve(data.time[q],flux = data.raw_flux[q]/np.median(data.raw_flux[q]), flux_err = data.flux_err[q], targetid = target_ID)
    if plot_raw == True:
        raw_eleanor_fig = plt.figure()
        plt.scatter(data.time[q], data.raw_flux[q]/np.median(data.raw_flux[q]), s=1, c = 'k')
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')
        plt.title('{} - eleanor light curve from FFIs - raw flux'.format(target_ID))
#        raw_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor raw flux.png'.format(target_ID, sector))
        #plt.close(raw_eleanor_fig)
        plt.show()
    
    # Plot corrected flux
    corr_lc = lightkurve.LightCurve(data.time[q],flux = data.corr_flux[q]/np.median(data.corr_flux[q]), flux_err = data.flux_err[q], targetid = target_ID)
    if plot_corr == True:
        corr_eleanor_fig = plt.figure()
        plt.scatter(data.time[q], data.corr_flux[q]/np.median(data.corr_flux[q]), s=1, c= 'r')
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')
        plt.title('{} - eleanor light curve from FFIs - corr flux'.format(target_ID))
        #corr_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor corr flux.png'.format(target_ID, sector))
        #plt.close(corr_eleanor_fig)
        plt.show()
    
    # Plot pca flux
    eleanor.TargetData.pca(data, flux=data.raw_flux, modes=4)
    pca_lc = lightkurve.LightCurve(data.time[q], flux = data.pca_flux[q]/np.median(data.pca_flux[q]), flux_err = data.flux_err[q], targetid = target_ID)
    if plot_pca == True:
        pca_eleanor_fig = plt.figure()
        plt.scatter(data.time[q], data.pca_flux[q]/np.median(data.pca_flux[q]), s=1, c= 'g')
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')
        plt.title('{} - eleanor light curve from FFIs - pca flux'.format(target_ID))
        pca_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor pca flux.png'.format(target_ID, sector))
        plt.close(pca_eleanor_fig)
        #plt.show()
    
    # Plot psf flux
    #eleanor.TargetData.psf_lightcurve(data, model='gaussian', likelihood='poisson')
    #psf_lc = lightkurve.LightCurve(data.time[q], flux = data.psf_flux[q]/np.median(data.psf_flux[q]), flux_err = data.flux_err[q], targetid = target_ID)
    #psf_eleanor_fig = plt.figure()
    #plt.scatter(data.time[q], data.psf_flux[q]/np.median(data.psf_flux[q]), s=1, c= 'g')
    #plt.ylabel('Normalized Flux')
    #plt.xlabel('Time')
    #plt.title('{} - eleanor light curve from FFIs - psf flux'.format(target_ID))
    #psf_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor psf flux.png'.format(target_ID, sector))
    #plt.show()
    
    return raw_lc, corr_lc, pca_lc #, psf_lc

def lc_from_csv(filename):
#    data = Table.read(filename, format='ascii.csv')
    data = pd.read_csv(filename)
    time = data['time']
    detrended_flux = data['flux'] 
    flux_err = data['flux_err']
    detrended_lc = lightkurve.LightCurve(time,detrended_flux, flux_err)
    return detrended_lc

def get_lc_from_fits(filename, source='QLP', clean=True):
    """
    Obtains lc from fits file if not one of the standard TESS pipelines
    n.b. Currently set up to handle QLP and CDIPS files, though easily extended
    """
#    fits.info(filename)
    hdul = fits.open(filename)
    data = hdul[1].data
#    print(hdul[1].header)
    
    if source == 'QLP':
        tic = filename[28:40]
        
        time = data['TIME']
        sap_flux = data['SAP_FLUX']
        kspsap_flux = data['KSPSAP_FLUX']
        quality = data['QUALITY']
#        quality = quality.astype(bool)
        quality = np.array([item>4095 for item in quality])
        
        if clean == True:
            clean_time = time[~quality]
            clean_flux = sap_flux[~quality]
            clean_kspsap_flux = kspsap_flux[~quality]
            quality = quality[~quality]
        else:
            clean_time = time
            clean_flux = sap_flux
            clean_kspsap_flux = kspsap_flux
            quality = quality
        
#        plt.figure()
#        plt.scatter(clean_time, clean_flux,s=1,c='k')
#        plt.title('SAP lc for TIC {}'.format(tic))
#        plt.show()
        
#        plt.figure()
#        plt.scatter(clean_time, clean_kspsap_flux,s=1,c='k')
#        plt.title('KSPSAP lc for TIC {}'.format(tic))
#        plt.show()
        
        hdul.close()
        
        lc = lightkurve.LightCurve(time = clean_time, flux = clean_flux, flux_err = quality, targetid = tic)
        return lc
    elif source == 'CDIPS':
#        print('Using CDIPS')
        tic = hdul[0].header['TICID']
        sector = hdul[0].header['SECTOR']
        
        time = data['TMID_BJD']
        time = [x - 2457000 for x in time]
        mag = data['IRM3']
        pca_mag = data['PCA3']
        mag_err = data['IRE3']
        
        
        plt.figure()
        plt.scatter(time, mag,s=1,c='k')
        plt.title('SAP lc for TIC {}'.format(tic))
        plt.show()
        
        plt.figure()
        plt.scatter(time, pca_mag,s=1,c='k')
        plt.title('PCA lc for TIC {}'.format(tic))
        plt.show()
        
        hdul.close()
        lc = lightkurve.LightCurve(time = time, flux = mag, flux_err = mag_err, targetid = tic)
        return lc, sector
    elif source == 'K2SFF':
        # Fill in K2SFF method
        k2sff_data = hdul['BESTAPER'].data
        k2sff_time = k2sff_data['T'] + 2454833 #Convert to BJD for consistency
        k2sff_flux = k2sff_data['FCOR']
        
        plt.figure()
        plt.scatter(k2sff_time,k2sff_flux,s=2,c='r', label = 'K2-SFF 30min')
        plt.legend()
        plt.xlabel('Time [BJD]')
        plt.ylabel("Normalized Flux")
        plt.show()
        hdul.close()
        lc = lightkurve.LightCurve(time = k2sff_time, flux = k2sff_flux)
        return lc
    else:
        print('Please enter a valid source')
        return 