#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:53:07 2020

@author: mbattley
"""

import os
import glob
from astropy.io import ascii
from astropy.table import Table
#from fits_handling import get_lc_from_fits

file_location = '/Users/mbattley/Documents/PhD/Python/ngts_test_folder/'
save_path = '/Users/mbattley/Documents/PhD/Python/'
#file_location = '/tess/photometry/tessFFIextract/lightcurves/'
#save_path = '/home/u1866052/'

tic_table = Table({'TIC':[],'Sector':[],'Camera':[],'ccd':[]},names=['TIC','Sector','Camera','ccd'])
tic_table['TIC'] = tic_table['TIC'].astype(str)
tic_table['Sector'] = tic_table['Sector'].astype(str)
tic_table['Camera'] = tic_table['Camera'].astype(str)
tic_table['ccd'] = tic_table['ccd'].astype(str)

#tic_list = []
#
#fileset = [file for file in glob.glob(file_location + "**/*.fits", recursive=True)]
#
#for file in fileset:
##    print(file)
##    print('TIC = {}'.format(file[-16:-5]))
##    print('Sector = {}'.format(file[-22]))
##    print('Camera = {}'.format(file[-24]))
##    print('Camera = {}'.format(file[-27:-25]))
#    tic = file[-16:-5]
#    sector = file[-22]
#    camera = file[-24]
#    ccd = file[-27:-25]
#    tic_table.add_row([tic,sector,camera,ccd])

#ascii.write(tic_table, save_path + 'tessFFIextract_TIC_list.csv', format='csv', overwrite = True)

#with open("tessFFIextract_tic_list.txt", "w") as output:
#    output.write(str(tic_list))
#


f = open('tessFFIextract_TIC_list_S1.csv','w')
f.write("TIC,Sector,Camera,ccd\n")

file_dirs = ['S01_1-1','S01_1-2','S01_1-3','S01_1-4','S01_2-1','S01_2-2','S01_2-3','S01_2-4','S01_3-1','S01_3-2','S01_3-3','S01_3-4','S01_4-1','S01_4-2','S01_4-3','S01_4-4']
for file_dir in file_dirs:
    for root,dirs,files in os.walk(file_location+file_dir+'/', topdown=True): 
#for root,dirs,files in os.walk(file_location, topdown=True):
        for item in files:
          if '.fits' in item:
             print(root)
             print('TIC = {}'.format(item[4:-5]))
    #             print('Sector = {}'.format(int(root[-7:-5])))
    #             print('Camera = {}'.format(root[-4]))
    #             print('ccd = {}'.format(root[-2]))
             tic = item[4:-5]
             sector = int(root[-7:-5]) 
             camera = root[-4]
             ccd = root[-2]
    #         tic_table.add_row([tic,sector,camera,ccd])
             f.write("{},{},{},{}\n".format(tic,sector,camera,ccd))
#    print(root) 
#    print(dirs) 
#    print files 
#    print('***************')
f.close()
#ascii.write(tic_table, save_path + 'tessFFIextract_TIC_list_S1.csv', format='csv', overwrite = True)
#directory_str = '/Users/mbattley/Documents/PhD/Python/CDIPS_lcs'
#directory = os.fsencode(directory_str)
#
#tic_list = []
#
#for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".fits"): 
##         lc = get_lc_from_fits(directory_str+'/'+filename,'CDIPS')
#         tic_list.append(filename[4:13])
#         continue
#     else:
#         continue