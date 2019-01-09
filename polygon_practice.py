# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:33:03 2018
This script practices the use of polygons - both for drawing/masking and 
selecting data.

@author: MatthewTemp
"""

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

# Defines vertices for polygon (path)
verts = [
        (-1.,-1.),
        (-1.,1.),
        (1.,1.),
        (1.,-1.),
        (0.,0.),
        (-1.,-1.)
        ]

#codes = [Path.MOVETO,
#         Path.LINETO,
#         Path.LINETO,
#         Path.LINETO,
#         Path.LINETO,
#         Path.CLOSEPOLY,
#         ]

# Sets up path
path = Path(verts)

# Generates random data between 2 and -2
x = np.random.randn(5)
y = np.random.randn(5)
points = np.column_stack((x,y))

# Plots data and polygon
fig = plt.figure()
plt.scatter(x,y)
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, lw=2, fill = False, color = 'r')
ax.add_patch(patch)
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
plt.show()

# Determines which data is inside polygon
inside = path.contains_points(points)
print(inside)