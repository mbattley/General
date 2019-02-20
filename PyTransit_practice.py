#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:39:53 2019

Script to practice using the PyTransit Tool

@author: phrhzn
"""

import numpy as np
from pytransit import MandelAgol
import matplotlib.pyplot as plt

t = np.linspace(0.8,1.2,500)
k, t0, p, a, i, e, w = 0.1, 1.01, 4, 8, 0.48*np.pi, 0.2, 0.5*np.pi
u = [0.25,0.10]

m = MandelAgol()
f = m.evaluate(t, k, u, t0, p, a, i, e, w)

plt.plot(t,f)