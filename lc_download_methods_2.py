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
from astropy.table import Table
from astropy.coordinates import SkyCoord
from lightkurve import search_lightcurvefile
from numpy import loadtxt
from astroquery.mast import Tesscut
from astroquery.mast import Catalogs

def find_tic(target_ID, from_file = True):
    if from_file == True:
        try:
            table_data = Table.read("Original_BANYAN_XI-III_xmatch_TIC.csv" , format='ascii.csv')
            #table_data = Table.read("Original VCA Members.csv" , format='ascii.csv') 
            #table_data = Table.read("Original Argus members info.csv" , format='ascii.csv')
            
            # Obtains ra and dec for object from target_ID
            i = list(table_data['main_id']).index(target_ID)
            ra = table_data['ra'][i]
            dec = table_data['dec'][i]
            tic = table_data['MatchID'][i]
        except:
            try:
                TIC_table = Catalogs.query_object(target_ID, catalog = "TIC")
                ra = TIC_table['ra'][0]
                dec = TIC_table['dec'][0]
                tic = TIC_table['ID'][0] 
            except:
                table_data = Table.read('BANYAN_XI-III_combined_members.csv')
                i = list(table_data['main_id']).index(target_ID)
                ra = table_data['ra'][i]
                dec = table_data['dec'][i]
                object_coord = SkyCoord(ra, dec, unit="deg")
                TIC_table = Catalogs.query_region(object_coord, radius = '1 deg', catalog = 'TIC')
                tic = TIC_table['ID'][0]
    else:
        # Find ra, dec and tic # via the TIC (typically based on Gaia DR2)
        try:
            TIC_table = Catalogs.query_object(target_ID, catalog = "TIC")
            ra = TIC_table['ra'][0]
            dec = TIC_table['dec'][0]
            tic = TIC_table['ID'][0] 
        except:
            table_data = Table.read('BANYAN_XI-III_combined_members.csv')
            i = list(table_data['main_id']).index(target_ID)
            ra = table_data['ra'][i]
            dec = table_data['dec'][i]
            object_coord = SkyCoord(ra, dec, unit="deg")
            TIC_table = Catalogs.query_region(object_coord, radius = '1 deg', catalog = 'TIC')
            tic = TIC_table['ID'][0]
    
    return ra, dec, tic
    

def two_min_lc_download(target_ID, sector, plt_SAP = False, plt_PDCSAP = False):
    """
    Downloads and returns SAP and PDCSAP 2-min lightcurves from MAST
    """
    try:
        dec, tic = find_tic(target_ID, from_file = True)
        lcf = search_lightcurvefile(tic, sector = sector).download()
    except:
        table_data = Table.read('BANYAN_XI-III_combined_members.csv')
        i = list(table_data['main_id']).index(target_ID)
        ra = table_data['ra'][i]
        dec = table_data['dec'][i]
        object_coord = SkyCoord(ra, dec, unit="deg")
        lcf = search_lightcurvefile(object_coord, sector = sector).download()
    
    # Seperate lightcurves
    sap_lc = lcf.SAP_FLUX
    pdcsap_lc = lcf.PDCSAP_FLUX
    if plt_SAP == True:
        pdcsap_lc.scatter()
        plt.title('{} - 2min SAP lc'.format(target_ID))
        plt.xlabel("Time - 2457000 (BTJD days)")
        plt.ylabel("Relative flux")
        plt.show()
    if plt_PDCSAP == True:
        pdcsap_lc.scatter()
        plt.title('{} - 2min PDCSAP lc'.format(target_ID))
        plt.xlabel("Time - 2457000 (BTJD days)")
        plt.ylabel("Relative flux")
        plt.show()
    #lcf.close()
    
    return sap_lc, pdcsap_lc


def raw_FFI_lc_download(target_ID, sector, plot_tpf = False, plot_lc = False, save_path = '', from_file = False):
    """
    Downloads and returns 30min cadence lightcurves based on SAP analysis of 
    the raw FFIs
    """    
    if from_file == True:
        with open('Sector_{}_target_filenames.pkl'.format(sector), 'rb') as f:
            target_filenames = pickle.load(f)
        f.close()
    else:
        target_filenames = {}
    
        ra, dec, tic = find_tic(target_ID, from_file = True) 
         
        object_coord = SkyCoord(ra, dec, unit="deg")
        manifest = Tesscut.download_cutouts(object_coord, [11,11], path = './TESS_Sector_1_cutouts')
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

def diff_image_lc_download(target_ID, sector, plot_lc = False, from_file = True, save_path = '/home/u1866052/Lowess detrending/TESS S5/'):
    """
    Downloads and returns 30min cadence lightcurves based on Oelkers & Stassun
    difference imaging analysis method of lightcurve extraction
    """
    DIAdir = '/ngts/scratch/tess/FFI-LC/S{}/lc/clean/'.format(sector)
    
    ra, dec, tic = find_tic(target_ID, from_file = from_file) 
    
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
    
    filename = '{}_sector0{}_{}_{}.lc'.format(tic, sector, camera, ccd)
#    filename = '{}_sector0{}_{}_{}.lc'.format(tic, sector, camera, ccd)
#    filename = '410214986_sector01_3_2.lc'

    
    try:
        #lines = loadtxt(filename, delimiter = ' ') # For when in local directory
        lines = loadtxt(DIAdir+filename, delimiter = ' ') # For when on ngtshead
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
#            diffImage_fig.savefig(save_path + '{} - Sector {} - DiffImage flux.png'.format(target_ID, sector))
#            plt.close(diffImage_fig)
            plt.show()
        
        lc = lightkurve.LightCurve(time = DIA_lc[0],flux = norm_flux, flux_err = DIA_lc[2], targetid = target_ID)
        
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
    ra, dec, tic = find_tic(target_ID, from_file = from_file) 
    
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
        #pca_eleanor_fig.savefig(save_path + '{} - Sector {} - eleanor pca flux.png'.format(target_ID, sector))
        #plt.close(pca_eleanor_fig)
        plt.show()
    
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
    
    return raw_lc, corr_lc, pca_lc #,psf_lc

