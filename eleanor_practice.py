#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:47:43 2019

Script to use the Eleanor tool to interact with TESS FFIs

@author: phrhzn
"""

import eleanor
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import astropy.units as u

from astropy.stats import BoxLeastSquares, sigma_clip
from IPython.display import Image
from scipy import signal
from scipy.interpolate import interp1d
from copy import copy, deepcopy

def plot_lc(x,y,title,xlabel = 'Time', ylabel = 'Normalized Flux', folded = False):
    """
    Plots lightcurve for specific x, y, and title
    """
    plt.figure()
    plt.plot(x, y, '.')
    plt.ylabel(ylabel)
    if folded == True:
        plt.xlabel('Phase')
    else:
        plt.xlabel(xlabel)
    plt.title(title)
    
    
#def plot_tpf(data, ax=None, frame=0, cadenceno=None, bkg=False, aperture_mask=None,
#         show_colorbar=True, mask_color='pink', style='lightkurve', **kwargs):
#    """Plot the pixel data for a single frame (i.e. at a single time).
#
#    The time can be specified by frame index number (`frame=0` will show the
#    first frame) or absolute cadence number (`cadenceno`).
#
#    Parameters
#    ----------
#    ax : matplotlib.axes._subplots.AxesSubplot
#        A matplotlib axes object to plot into. If no axes is provided,
#        a new one will be generated.
#    frame : int
#        Frame number. The default is 0, i.e. the first frame.
#    cadenceno : int, optional
#        Alternatively, a cadence number can be provided.
#        This argument has priority over frame number.
#    bkg : bool
#        If True, background will be added to the pixel values.
#    aperture_mask : ndarray or str
#        Highlight pixels selected by aperture_mask.
#    show_colorbar : bool
#        Whether or not to show the colorbar
#    mask_color : str
#        Color to show the aperture mask
#    style : str
#        Path or URL to a matplotlib style file, or name of one of
#        matplotlib's built-in stylesheets (e.g. 'ggplot').
#        Lightkurve's custom stylesheet is used by default.
#    kwargs : dict
#        Keywords arguments passed to `lightkurve.utils.plot_image`.
#
#    Returns
#    -------
#    ax : matplotlib.axes._subplots.AxesSubplot
#        The matplotlib axes object.
#    """
#    if style == 'lightkurve' or style is None:
#        style = MPLSTYLE
#    if cadenceno is not None:
#        try:
#            frame = np.argwhere(cadenceno == self.cadenceno)[0][0]
#        except IndexError:
#            raise ValueError("cadenceno {} is out of bounds, "
#                             "must be in the range {}-{}.".format(
#                                 cadenceno, self.cadenceno[0], self.cadenceno[-1]))
#    try:
#        if bkg and np.any(np.isfinite(self.flux_bkg[frame])):
#            pflux = self.flux[frame] + self.flux_bkg[frame]
#        else:
#            pflux = self.flux[frame]
#    except IndexError:
#        raise ValueError("frame {} is out of bounds, must be in the range "
#                         "0-{}.".format(frame, self.shape[0]))
#    with plt.style.context(style):
#        img_title = 'Target ID: {}'.format(self.targetid)
#        img_extent = (self.column, self.column + self.shape[2],
#                      self.row, self.row + self.shape[1])
#        ax = plot_image(pflux, ax=ax, title=img_title, extent=img_extent,
#                        show_colorbar=show_colorbar, **kwargs)
#        ax.grid(False)
#    if aperture_mask is not None:
#        aperture_mask = self._parse_aperture_mask(aperture_mask)
#        for i in range(self.shape[1]):
#            for j in range(self.shape[2]):
#                if aperture_mask[i, j]:
#                    ax.add_patch(patches.Rectangle((j+self.column, i+self.row),
#                                                   1, 1, color=mask_color, fill=True,
#                                                   alpha=.6))
#    return ax
    
def make_transit_periodogram(t,y,dy=0.01):
    """
    Plots a periodogram to determine likely period of planet transit candidtaes 
    in a dataset, based on a box least squared method.
    """
    model = BoxLeastSquares(t * u.day, y, dy = 0.01)
    periodogram = model.autopower(0.2, objective="snr")
    plt.figure()
    plt.plot(periodogram.period, periodogram.power,'k')
    plt.xlabel('Period [days]')
    plt.ylabel('Power')
    max_power_i = np.argmax(periodogram.power)
    best_fit = periodogram.period[max_power_i]
    print('Best Fit Period: {} days'.format(best_fit))
    stats = model.compute_stats(periodogram.period[max_power_i],
                                periodogram.duration[max_power_i],
                                periodogram.transit_time[max_power_i])
    return best_fit, stats

def fold(data, period, transit_midpoint=0.):
    """Folds the lightcurve at a specified ``period`` and ``transit_midpoint``.

    This method returns a new ``LightCurve`` object in which the time
    values range between -0.5 to +0.5 (i.e. the phase).
    Data points which occur exactly at ``transit_midpoint`` or an integer
    multiple of `transit_midpoint + n*period` will have time value 0.0.

    Parameters
    ----------
    period : float
        The period upon which to fold.
    transit_midpoint : float, optional
        Time reference point in the same units as the LightCurve's `time`
        attribute.

    Returns
    -------
    folded_lightcurve : LightCurve object
        A new ``LightCurve`` in which the data are folded and sorted by
        phase.
    """
    phase = (transit_midpoint % period) / period
    fold_time = (((data.time - phase * period) / period) % 1)
    # fold time domain from -.5 to .5
    fold_time[fold_time > 0.5] -= 1
    sorted_args = np.argsort(fold_time)
    data.time=fold_time[sorted_args]
    data.raw_flux = data.raw_flux[sorted_args]
    data.flux_err = data.flux_err[sorted_args]
    return data

def flatten(data, window_length=1001, polyorder=2, return_trend=False,
            break_tolerance=None, niters=3, sigma=3, mask=None, **kwargs):
    """Removes the low frequency trend using scipy's Savitzky-Golay filter.

    This function wraps `scipy.signal.savgol_filter` and is based on the 
    method from lightkurve.
    
    Parameters
        ----------
        window_length : int
            The length of the filter window (i.e. the number of coefficients).
            ``window_length`` must be a positive odd integer.
        polyorder : int
            The order of the polynomial used to fit the samples. ``polyorder``
            must be less than window_length.
        return_trend : bool
            If `True`, the method will return a tuple of two elements
            (flattened_lc, trend_lc) where trend_lc is the removed trend.
        break_tolerance : int
            If there are large gaps in time, flatten will split the flux into
            several sub-lightcurves and apply `savgol_filter` to each
            individually. A gap is defined as a period in time larger than
            `break_tolerance` times the median gap.  To disable this feature,
            set `break_tolerance` to None.
        niters : int
            Number of iterations to iteratively sigma clip and flatten. If more than one, will
            perform the flatten several times, removing outliers each time.
        sigma : int
            Number of sigma above which to remove outliers from the flatten
        mask : boolean array with length of self.time
            Boolean array to mask data with before flattening. Flux values where
            mask is True will not be used to flatten the data. An interpolated
            result will be provided for these points. Use this mask to remove
            data you want to preserve, e.g. transits.
        **kwargs : dict
            Dictionary of arguments to be passed to `scipy.signal.savgol_filter`.
    
    """
    if mask is None:
        mask = np.ones(len(data.time), dtype=bool)
    else:
        # Deep copy ensures we don't change the original.
        mask = deepcopy(~mask)
    # No NaNs
    mask &= np.isfinite(data.raw_flux)
    # No outliers
    mask &= np.nan_to_num(np.abs(data.raw_flux - np.nanmedian(data.raw_flux))) <= (np.nanstd(data.raw_flux) * sigma)
    
    for iter in np.arange(0,niters):
        if break_tolerance is None:
            break_tolerance = np.nan
        if polyorder >= window_length:
            polyorder = window_length - 1
            log.warning("polyorder must be smaller than window_length, "
                        "using polyorder={}.".format(polyorder))
        # Split the lightcurve into segments by finding large gaps in time
        dt = data.time[mask][1:] - data.time[mask][0:-1]
        with warnings.catch_warnings():  # Ignore warnings due to NaNs
            warnings.simplefilter("ignore", RuntimeWarning)
            cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
        low = np.append([0], cut)
        high = np.append(cut, len(data.time[mask]))
        # Then, apply the savgol_filter to each segment separately
        trend_signal = np.zeros(len(data.time[mask]))
        for l, h in zip(low, high):
            # Reduce `window_length` and `polyorder` for short segments;
            # this prevents `savgol_filter` from raising an exception
            # If the segment is too short, just take the median
            if np.any([window_length > (h - l), (h - l) < break_tolerance]):
                trend_signal[l:h] = np.nanmedian(data.raw_flux[mask][l:h])
            else:
                # Scipy outputs a warning here that is not useful, will be fixed in version 1.2
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    trend_signal[l:h] = signal.savgol_filter(x=data.raw_flux[mask][l:h],
                                                             window_length=window_length,
                                                             polyorder=polyorder,
                                                             **kwargs)
        # No outliers
        mask1 = np.nan_to_num(np.abs(data.raw_flux[mask] - trend_signal)) <\
                (np.nanstd(data.raw_flux[mask] - trend_signal) * sigma)
        f = interp1d(data.time[mask][mask1], trend_signal[mask1], fill_value='extrapolate')
        trend_signal = f(data.time)
        mask[mask] &= mask1
        
    flattened_data = copy(data)
    with warnings.catch_warnings():
        # ignore invalid division warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        flattened_data.raw_flux = data.raw_flux / trend_signal
        flattened_data.flux_err = data.flux_err / trend_signal
    if return_trend:
        trend_data = copy(data)
        trend_data.raw_flux = trend_signal
        return flattened_data, trend_data
    return flattened_data

def remove_outliers(data, sigma=5., return_mask=False, **kwargs):
    """Removes outlier data points using sigma-clipping.

    This method returns a new :class:`LightCurve` object from which data
    points are removed if their flux values are greater or smaller than
    the median flux by at least ``sigma`` times the standard deviation.

    Sigma-clipping works by iterating over data points, each time rejecting
    values that are discrepant by more than a specified number of standard
    deviations from a center value. If the data contains invalid values
    (NaNs or infs), they are automatically masked before performing the
    sigma clipping.

    .. note::
        This function is a convenience wrapper around
        `astropy.stats.sigma_clip
        <http://docs.astropy.org/en/stable/api/astropy.stats.sigma_clip.html>`_
        and provides the same functionality.

    Parameters
    ----------
    sigma : float
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 5.
    sigma_lower : float or `None`
        The number of standard deviations to use as the lower bound for
        the clipping limit. Can be set to float('inf') in order to avoid
        clipping outliers below the median at all. If `None` then the
        value of ``sigma`` is used. Defaults to `None`.
    sigma_upper : float or `None`
        The number of standard deviations to use as the upper bound for
        the clipping limit. Can be set to float('inf') in order to avoid
        clipping outliers above the median at all. If `None` then the
        value of ``sigma`` is used. Defaults to `None`.
    return_mask : bool
        Whether or not to return a mask (i.e. a boolean array) indicating
        which data points were removed. Entries marked as `True` in the
        mask are considered outliers. Defaults to `True`.
    iters : int or `None`
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    cenfunc : callable
        The function used to compute the center for the clipping. Must
        be a callable that takes in a masked array and outputs the
        central value. Defaults to the median (`numpy.ma.median`).
    **kwargs : dict
        Dictionary of arguments to be passed to `astropy.stats.sigma_clip`.

    Returns
    -------
    clean_data : data with outliers removed
    """
    # First, we create the outlier mask using AstroPy's sigma_clip function
    with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
        warnings.simplefilter("ignore")
        outlier_mask = sigma_clip(data=data.raw_flux, sigma=sigma, **kwargs).mask
    
    # Clip data
    data.raw_flux = data.raw_flux[~outlier_mask]
    data.time = data.time[~outlier_mask]
    data.flux_err = data.flux_err[~outlier_mask]
    
    # Second, we return the masked lightcurve and optionally the mask itself
    if return_mask:
        return data, outlier_mask
    return data

def bin(data, binsize=13, method='mean'):
    """Bins a lightcurve in blocks of size `binsize`.

    The value of the bins will contain the mean (`method='mean'`) or the
    median (`method='median'`) of the original data.  The default is mean.

    Parameters
    ----------
    binsize : int
        Number of cadences to include in every bin.
    method: str, one of 'mean' or 'median'
        The summary statistic to return for each bin. Default: 'mean'.

    Returns
    -------
    binned_lc : LightCurve object
        Binned lightcurve.

    Notes
    -----
    - If the ratio between the lightcurve length and the binsize is not
      a whole number, then the remainder of the data points will be
      ignored.
    - If the original lightcurve contains flux uncertainties (flux_err),
      the binned lightcurve will report the root-mean-square error.
      If no uncertainties are included, the binned curve will return the
      standard deviation of the data.
    """
    available_methods = ['mean', 'median']
    if method not in available_methods:
        raise ValueError("method must be one of: {}".format(available_methods))
    methodf = np.__dict__['nan' + method]

    n_bins = data.raw_flux[q].size // binsize
    binned_lc = copy(data)
    indexes = np.array_split(np.arange(len(data.time[q])), n_bins)
    binned_lc.time = np.array([methodf(data.time[q][a]) for a in indexes])
    binned_lc.raw_flux = np.array([methodf(data.raw_flux[q][a]) for a in indexes])

    if np.any(np.isfinite(data.flux_err[q])):
        # root-mean-square error
        binned_lc.flux_err = np.array(
            [np.sqrt(np.nansum(data.flux_err[q][a]**2))
             for a in indexes]
        ) / binsize
    else:
        # Make them zeros.
        binned_lc.flux_err = np.zeros(len(binned_lc.raw_flux[q]))

    return binned_lc

###############################################################################
# Main

log = logging.getLogger(__name__)

# Assign star of interest
#star = eleanor.Source(tic=29857954, sector=1)
star = eleanor.Source(coords=(319.94962, -58.1489), sector=1)
#star = eleanor.Source(gaia=4675352109658261376, sector=1)

# Extract target pixel file, perform aperture photometry and complete some systematics corrections
#data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False)
data = eleanor.TargetData(star)

q = data.quality == 0

# Plot a lightcurve or a few...
plt.figure()
plt.plot(data.time[q], data.raw_flux[q]/np.median(data.raw_flux[q])-0.01, 'k')
plt.plot(data.time[q], data.corr_flux[q]/np.median(data.corr_flux[q]) + 0.01, 'r')
#plt.plot(data.time[q], data.pca_flux[q]/np.median(data.pca_flux[q]) + 0.03, 'y')

plt.ylabel('Normalized Flux')
plt.xlabel('Time')
plt.show()

# View aperture
#plt.figure()
#plt.imshow(data.aperture)

#data.save(output_fn = 'tpf_from_ffi.fits', directory = '/Documents/PhD/Python')
#data.save(output_fn = 'tpf_from_ffi.fits')

## Create custom aperture
#eleanor.TargetData.custom_aperture(data, shape='circle', r=1)
#eleanor.TargetData.get_lightcurve(data)
#
# Create own mask
#mask = np.zeros(np.shape(data.tpf[0]), dtype = bool)
#mask[6:8,6:8] = 1
#plt.figure()
#plt.imshow(mask, origin='lower')
#data.get_lightcurve(aperture=mask)
#
## Create ultra-custom apertures
##In terminal: vis = eleanor.Visualize(target) cust_lc = vis.click_aperture()
##Image(url='customApExample.gif')
#
## Systematics corrections for custom aperture:
#corr_flux=eleanor.TargetData.jitter_corr(data, flux=data.raw_flux) # jitter
#pca_data = eleanor.TargetData.pca(data, flux=corr_flux, modes=4) # remove any shared systemtatics with nearby stars
##eleanor.TargetData.psf_lightcurve(data, model='gaussian', likelihood='poisson') #PSF modelling - needs tensorflow

# Plot new lightcurve
#plot_lc(data.time[q],data.raw_flux[q],'Original lightcurve for WASP-73', ylabel = 'Flux')

# Apply Savgol Filter
#flattened_data = flatten(data)

# Re-plot flattened data
#plot_lc(flattened_data.time[q],flattened_data.raw_flux[q],'Flattened lightcurve for WASP-73')

# Remove outliers via sigma clipping
#clean_data, outlier_mask = remove_outliers(flattened_data, sigma=5., return_mask = True)

#q = q[~outlier_mask]

# Re-plot cleaned data
#plot_lc(clean_data.time[q],clean_data.raw_flux[q],'Lightcurve for WASP-73 with outliers removed')

# Find optimum period
#best_period, stats = make_transit_periodogram(t = clean_data.time, y = clean_data.raw_flux)

# Fold data
#folded_data = fold(clean_data, period = 2.849375)
#folded_data = fold(clean_data, period = 2.84834994)
#folded_data = fold(clean_data, period = 4.08722)

# Re-plot folded data
#plot_lc(folded_data.time[q],folded_data.raw_flux[q],'Folded Lightcurve for WASP-73 b', folded = True)

# Bin data
#binned_data = bin(folded_data)

# Plot binned data
#plot_lc(binned_data.time,binned_data.raw_flux,'Folded Lightcurve for binned WASP-73 b', folded = True)