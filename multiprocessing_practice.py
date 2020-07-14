#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:27:56 2020

Multiprocessing practice

@author: mbattley
"""

import multiprocessing as multip
import time

start = time.time()

def cube(x):
    return x**3

print('Total number of processors on your machine is: {}'.format(multip.cpu_count()))

pool = multip.Pool(processes=1)
results = [pool.apply(cube, args=(x,)) for x in range(1,7)]
print(results)

end = time.time()
print('Elapsed time = {}s'.format(end - start))