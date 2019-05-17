#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:58:26 2019

@author: Matthew Battley
"""

from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from astropy.table import Table
import pickle
#import astropy.units as u

def download_TESS_cutouts_ID(target_ID_list, cutout_size = [11,11]):
    """
    This function downloads the TESS-cutouts for a specified list of target IDs
    and returns a dictionary of the associated filename(s) for each target
    
    Inputs:
        Name                    type              description
        -----------------------------------------------------------------------
        target_ID_ list         [n x 1] list      list of target IDs as strings
        cutout_size             [2 x 1] list      postcard cutout size
    
    Output:
        target_filenames         n x 2 dictionary with target ids and target filenames
    """

    filename = "BANYAN_XI-III_combined_members.csv"
    
    data = Table.read(filename, format='ascii.csv')
    
    target_filenames = {}
    
    for target_id in target_ID_list:
        i = list(data['main_id']).index(target_id)
        ra = data['ra'][i]
        dec = data['dec'][i]
        object_coord = SkyCoord(ra, dec, unit="deg")
        manifest = Tesscut.download_cutouts(object_coord, cutout_size, path = './TESS_Sector_1_cutouts')
        sector_info = Tesscut.get_sectors(object_coord)
        if len(manifest['Local Path']) == 1:
            target_filenames[target_id] = manifest['Local Path'][0][2:]
        elif len(manifest['Local Path']) > 1:
            target_filenames[target_id] = []
            for filename in manifest['Local Path']:
                target_filenames[target_id].append(filename[2:])
        else:
            print('Cutout for target {} can not be downloaded'.format(target_id))
    
    return target_filenames, sector_info

def download_TESS_cutouts_coords(ra, dec, cutout_size = [11,11]):
    """
    This function downloads the TESS-cutouts for a target with specified ra and
    dec coordinates and prints the resulting filename(s)
    
    Inputs:
        Name          unit           type                 description
        -----------------------------------------------------------------------
        ra            deg            float           right ascension of target
        dec           deg            float           declination of target
        cutout_size   ---            [2 x 1] list    postcard cutout size
    """
    cutout_coord = SkyCoord(ra, dec, unit="deg")
    manifest = Tesscut.download_cutouts(cutout_coord, cutout_size)
    print(manifest)

################################ Main #########################################

# Reload dictionary of YSO targets in each sector
with open('TESS_YSO_sector_dict.pkl', 'rb') as f:
    full_YSO_target_dict = pickle.load(f)

# Choose specific sector of interest and retrieve targets
target_ID_list = full_YSO_target_dict['S1']
target_filenames = download_TESS_cutouts_ID(target_ID_list)

with open('Sector_1_target_filenames.pkl', 'wb') as f:
    pickle.dump(target_filenames, f, pickle.HIGHEST_PROTOCOL)