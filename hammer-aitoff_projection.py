#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:51:29 2019

Script to create sky plots with Hammer-Aitoff projection, using star positions
from a file

@author: Matthew Battley
"""

import numpy as np
import astropy.coordinates as coord
import astropy.table as tab
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

file_location = '/Users/mbattley/Documents/PhD/Association Members/'

Table = tab.Table

full_assoc_list = ['118TAU','ABDMG','BPMG','CAR','CARN','CBER','COL','CRA','EPSC',
                   'ETAC','HYA','IC239','IC2602','LCC','OCT','PL8','PLE','ROPH','TAU',
                   'THA','THOR','TWA','UCL','UCRA','UMA','USCO','XFOR']

assoc_list1 = ['118TAU','ABDMG','BPMG','CAR','CARN','CBER','COL','CRA','EPSC','ETAC']
assoc_list2 = ['HYA','IC239','IC2602','LCC','OCT','PL8','PLE','ROPH','TAU','THA']
assoc_list3 = ['THOR','TWA','UCL','UCRA','UMA','USCO','XFOR']

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='aitoff')

for assoc in full_assoc_list:
    data = Table.read(file_location + '{} Confirmed BANYAN members with TIC.xml'.format(assoc))
    
    ra = coord.Angle(data['ra'].filled(np.nan)*u.degree)
    ra = ra.wrap_at(180*u.degree)
    dec = coord.Angle(data['dec'].filled(np.nan)*u.degree)
    if assoc in assoc_list1:
        ax.scatter(ra.radian, dec.radian, s = 2, label = assoc)
    elif assoc in assoc_list2:
        ax.scatter(ra.radian, dec.radian, s = 8, marker = '>', label = assoc, alpha = 0.5)
    else:
        ax.scatter(ra.radian, dec.radian, s = 4, marker = 's', label = assoc, alpha = 0.5)

verts = [
        (-30.*(np.pi/180),-30.*(np.pi/180)),
        (-30.*(np.pi/180),-60.*(np.pi/180)),
        (-60.*(np.pi/180),-60.*(np.pi/180)),
        (-60.*(np.pi/180),-30.*(np.pi/180)),
        (-30.*(np.pi/180),-30.*(np.pi/180))
        ]
path = Path(verts)

#plt.scatter([-30,-30,-60,-60,-30],[-30,-60,-60,-30,-30], s = 10, marker = 'h') 

ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
ax.grid(True)
ax.legend(bbox_to_anchor=(1.01,0.5), loc="right", bbox_transform=fig.transFigure, ncol=1)
plt.title('Distribution of young stars and their respective associations across the sky \n')
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, lw=2, fill = False, color = 'b')
ax.add_patch(patch)
plt.show()

########################### For single set of data: ###########################


#data = Table.read('Bona-fide Octans members with dist.vot')
#
#ra = coord.Angle(data['ra'].filled(np.nan)*u.degree)
#ra = ra.wrap_at(180*u.degree)
#dec = coord.Angle(data['dec'].filled(np.nan)*u.degree)
#
#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111, projection="aitoff")
#ax.scatter(ra.radian, dec.radian, s = 2)
##ax.scatter(ra.radian, dec.radian, s = 2, c = 'b', marker = '*')
#ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
#ax.grid(True)
#plt.title('Distribution of young stars and their respective associations across the sky \n')
#plt.show()