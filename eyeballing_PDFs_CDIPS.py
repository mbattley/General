#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:23:20 2020

Eyeballing_PDFs.py

Script to speed up the eyeballing process for a pre-defined list of pdf files.

@author: mbattley
"""
import subprocess
import csv
import signal
import os
from astropy.table import Table

save_path = "/home/astro/phrhzn/Desktop/cam4/"

data = Table.read(save_path + 'Period_info_table_roseta_part1.csv', format='ascii.csv')
TIC_list = data['TIC'][401:501]
#print(TIC_list)

#TIC_list = [6951741,68161800]
##TIC_list = [68161800]
#
with open(save_path + 'Eyeballing_notes_401-500.csv','w') as f:
    header_row = ['TIC','Eyeballing Notes']
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_row)

for tic in TIC_list:
#    full_file_name = r'file://C:' + file_path + 'TIC ' + str(tic) + ' Full eyeballing fig.pdf'
#    print(full_file_name)
    
    filename = "/Users/mbattley/Documents/PhD/New\ detrending\ methods/Smoothing/lowess/roseta_part1/TIC\ {}\ -\ Full\ eyeballing\ fig.pdf".format(tic[4:])
    p = subprocess.Popen('exec open -a Preview.app ' + filename,stdout=subprocess.PIPE, shell=True)
#    subprocess.run(['open',filename])
    
    print(tic)
    eyeballing_notes = input('Enter eyeballing notes:')
#    print('Hello, ' + eyeballing_notes)
    
    with open(save_path + 'Eyeballing_notes_401-500.csv','a') as f:
        data_row = [str(tic),eyeballing_notes]
        writer = csv.writer(f, delimiter=',')
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerow(data_row)
    
#    subprocess.call(['close',filename])
    p.kill()
#    os.killpg(os.getpgid(p.pid), signal.SIGTERM)