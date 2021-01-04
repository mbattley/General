#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:03:32 2020

lowess_detrend.py

Function for standard lowess-detrending step for use anywhere

@author: mbattley
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt



############################## LOWESS detrending ##############################

def lowess_detrending(time=[],flux=[],target_ID='',pipeline='2min',detrending='lowess_partial',n_bins=30,save_path=''):

    # Full lc
    if detrending == 'lowess_full':
        full_lowess_flux = np.array([])
        lowess = sm.nonparametric.lowess(flux, time, frac=0.02)
        
    #     number of points = 20 at lowest, or otherwise frac = 20/len(t_section) 
#        print(lowess)
        overplotted_lowess_full_fig = plt.figure()
        plt.scatter(time,flux, c = 'k', s = 1)
        plt.plot(lowess[:, 0], lowess[:, 1])
        plt.title('{} lc with overplotted lowess full lc detrending'.format(target_ID))
        plt.xlabel('Time [BJD days]')
        plt.ylabel('Relative flux')
        #overplotted_lowess_full_fig.savefig(save_path + "{} lc with overplotted LOWESS full lc detrending.png".format(target_ID))
        plt.show()
    #   plt.close(overplotted_lowess_full_fig)
        
        residual_flux_lowess = flux/lowess[:,1]
        full_lowess_flux = np.concatenate((full_lowess_flux,lowess[:,1]))
        
        lowess_full_residuals_fig = plt.figure()
        plt.scatter(time,residual_flux_lowess, c = 'k', s = 1)
        plt.title('{} lc after lowess full lc detrending'.format(target_ID))
        plt.xlabel('Time [BJD days]')
        plt.ylabel('Relative flux')
        ax = plt.gca()
        #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #            lowess_full_residuals_fig.savefig(save_path + "{} lc after LOWESS full lc detrending.png".format(target_ID))
        plt.show()
    #                plt.close(lowess_full_residuals_fig)
        
        
    # Partial lc
    elif detrending == 'lowess_partial':
        expand_final = False
        time_diff = np.diff(time)
        residual_flux_lowess = np.array([])
        time_from_lowess_detrend = np.array([])
        full_lowess_flux = np.array([])
        
        overplotted_detrending_fig = plt.figure()
        plt.scatter(time,flux, c = 'k', s = 2)
        plt.xlabel('Time [BJD days]')
        plt.ylabel("Normalized flux")
        #plt.title('{} lc with overplotted detrending'.format(target_ID))
        
        low_bound = 0
        if pipeline == '2min':
            n_bins = 15*n_bins
        else:
            n_bins = n_bins
        for i in range(len(time)-1):
            if time_diff[i] > 0.1:
                high_bound = i+1
                
                t_section = time[low_bound:high_bound]
                flux_section = flux[low_bound:high_bound]
    #                        print(t_section)
                if len(t_section)>=n_bins:
                    lowess = sm.nonparametric.lowess(flux_section, t_section, frac=n_bins/len(t_section))
    #                    lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
                    lowess_flux_section = lowess[:,1]
                    plt.plot(t_section, lowess_flux_section, '-')
                    
                    residuals_section = flux_section/lowess_flux_section
                    residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
                    time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
                    full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section))
                    low_bound = high_bound
                else:
                    print('Skipped one gap at {}'.format(high_bound))
        
        # Carries out same process for final line (up to end of data)        
        high_bound = len(time)
        
        if high_bound - low_bound < n_bins:
            old_low_bound = low_bound
            low_bound = high_bound - n_bins
            expand_final = True
        t_section = time[low_bound:high_bound]
        flux_section = flux[low_bound:high_bound]
        lowess = sm.nonparametric.lowess(flux_section, t_section, frac=n_bins/len(t_section))
    #            lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
        lowess_flux_section = lowess[:,1]
        plt.plot(t_section, lowess_flux_section, '-')
    #                plt.title('AU Mic - Overplotted LOWESS detrending')
        overplotted_detrending_fig.savefig(save_path + "{} - Overplotted lowess detrending - partial lc".format(target_ID))
        overplotted_detrending_fig.show()
    #                plt.close(overplotted_detrending_fig)
        
        residuals_section = flux_section/lowess_flux_section
        if expand_final == True:
            shorten_bound = n_bins - (high_bound-old_low_bound)
            residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section[shorten_bound:]))
            time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section[shorten_bound:]))
            full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section[shorten_bound:]))
        else:
            residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
            time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
            full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section))
        
    #    t_section = t_cut[83:133]
        residuals_after_lowess_fig = plt.figure()
        plt.scatter(time_from_lowess_detrend,residual_flux_lowess, c = 'k', s = 2)
        plt.title('{} lc after LOWESS partial lc detrending'.format(target_ID))
        plt.xlabel('Time - 2457000 [BTJD days]')
        plt.ylabel('Relative flux')
        #ax = plt.gca()
        #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
        residuals_after_lowess_fig.savefig(save_path + "{} lc after LOWESS partial lc detrending".format(target_ID))
        residuals_after_lowess_fig.show()
    #                plt.close(residuals_after_lowess_fig)
    
    return residual_flux_lowess, full_lowess_flux