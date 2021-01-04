#!/usr/bin/env python

"""
"""
    
import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit
from astropy import coordinates, units as u
import mgutils as mg, mgutils.constants as co



if "-h" in argv:
    print ("footprint.py usage")
    raise SystemExit


fname = 'footprint.csv'


def plot_TESS_footprint(fname, ax_ecliptic_coords, ax_radec_coords, camera_1_fmt='r-', camera_2_fmt='m-', camera_3_fmt='b-', camera_4_fmt='g-', *args, **kwargs ):
    """ Function to plot the TESS footprint. 'fname' is the path to a csv file containing coordinates of the 4 cameras per sector
    """

    # Read in RA and Dec of each camera
    ra1, dec1, ra2, dec2, ra3, dec3, ra4, dec4, = np.loadtxt(fname, delimiter=',', unpack=True, usecols=(5,6,8,9,11,12,14,15))

    # Loop over sectors
    for j in range(26):
        ## Camera 1
        plot_edges(ax_ecliptic_coords, ax_radec_coords, ra1[j], dec1[j], camera_1_fmt, *args, **kwargs)
        ## Camera 2
        plot_edges(ax_ecliptic_coords, ax_radec_coords, ra2[j], dec2[j], camera_2_fmt, *args, **kwargs)
        ## Camera 3
        plot_edges(ax_ecliptic_coords, ax_radec_coords, ra3[j], dec3[j], camera_3_fmt, *args, **kwargs)
        ## Camera 4 (looks funky?)
        # plot_edges(ax_ecliptic_coords, ax_radec_coords, ra4[j], dec4[j], camera_4_fmt, *args, **kwargs)


def plot_edges(ax_ecliptic_coords, ax_radec_coords, ra, dec, *args, **kwargs):
    """ Generate coords of frame edges using find_edges function, then plot them
        Will plot one axis with ecliptic coords and one with RA/dec coords
        If you don't want one of those plots, set the corresponding input axis object to None
        Currently assumes the ecliptic plot is in a Cartesian projection, and the RA/Dec plot is in a Mollweide projection.
        Any extra arguments will be passed to plt.plot()
    """
    # Create edge coordinates
    edge_left, edge_top, edge_right, edge_bottom = find_edges(ra, dec)

    # Plot in ecliptic coordinates
    if ax_ecliptic_coords is not None:
        ax_ecliptic_coords.plot(edge_left.lon, edge_left.lat, *args, **kwargs)
        ax_ecliptic_coords.plot(edge_top.lon, edge_top.lat, *args, **kwargs)
        ax_ecliptic_coords.plot(edge_right.lon, edge_right.lat, *args, **kwargs)
        ax_ecliptic_coords.plot(edge_bottom.lon, edge_bottom.lat, *args, **kwargs)

    # Plot in equatorial (RA/Dec) coordinates
    if ax_radec_coords is not None:
        safe_plot(edge_left, ax_radec_coords, *args, **kwargs)
        safe_plot(edge_top, ax_radec_coords, *args, **kwargs)
        safe_plot(edge_right, ax_radec_coords, *args, **kwargs)
        safe_plot(edge_bottom, ax_radec_coords, *args, **kwargs)



def find_edges(ra, dec):
    """ Given the RA and Dec of the centre of a TESS camera, calculate the coordinates of the edges of the camera frame.
        Does this by converting to ecliptic, assuming a square 24x24 deg camera that is orthogonal to ecliptic plane
    """
    ### Convert coordinates of camera centre to ecliptic coords
    ecl_frame = coordinates.GeocentricMeanEcliptic
    coords = coordinates.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    ecl_coords = coords.transform_to(ecl_frame)
    eclon, eclat = ecl_coords.lon, ecl_coords.lat

    # useful arrays
    ran = np.linspace(-12,+12,100)
    twelves = 12 * np.ones(len(ran))

    ### Create arrays of coords representing each edge of the camera
    edge_left = coordinates.SkyCoord(*wrap_coords(eclon - twelves*u.deg, eclat + ran*u.deg), frame=ecl_frame)
    edge_top = coordinates.SkyCoord(*wrap_coords(eclon + ran*u.deg, eclat + twelves*u.deg), frame=ecl_frame)
    edge_right = coordinates.SkyCoord(*wrap_coords(eclon + twelves*u.deg, eclat - ran*u.deg), frame=ecl_frame)
    edge_bottom = coordinates.SkyCoord(*wrap_coords(eclon - ran*u.deg, eclat - twelves*u.deg), frame=ecl_frame)

    return edge_left, edge_top, edge_right, edge_bottom


def wrap_coords(lon, lat):
    """ Handles wrapping coords that would otherwise have lat > 90 degrees
    """
    if (np.abs(lat) < 90*u.deg).all():
        return lon, lat

    # offset longitude by 180 deg
    lon[np.abs(lat) > 90*u.deg] = (180*u.deg + lon[np.abs(lat) > 90*u.deg]) % (360*u.deg)
    
    # wrap latitude around
    lat[lat > 90*u.deg] = 180*u.deg - lat[lat > 90*u.deg]
    lat[lat < -90*u.deg] = -180*u.deg - lat[lat < -90*u.deg]
    return lon, lat
    


def safe_plot(edge, ax, *args, **kwargs):
    """ A quick and dirty fix to plot lines that wrap around the join at 180 degrees without getting a load of ugly horizontal lines
        NB - this function plots in radians to match what matplotlib expects for the "mollweide" projection
    """
    if (edge.icrs.ra<180*u.deg).any():
        ax.plot(edge.icrs.ra.radian[edge.icrs.ra<180*u.deg], edge.icrs.dec.radian[edge.icrs.ra<180*u.deg], *args, **kwargs)
    if (edge.icrs.ra>180*u.deg).any():
        ax.plot(edge.icrs.ra.wrap_at(180*u.deg).radian[edge.icrs.ra>180*u.deg], edge.icrs.dec.radian[edge.icrs.ra>180*u.deg], *args, **kwargs)




if __name__ == "__main__":

    # generate figures
    fig1 = plt.figure(1)
    ax_ecliptic_coords = fig1.add_subplot(111)
    fig2 = plt.figure(2)
    ax_radec_coords = fig2.add_subplot(111, projection="mollweide")
    ax_radec_coords.grid(True)

    # plot
    plot_TESS_footprint(fname, ax_ecliptic_coords, ax_radec_coords, camera_1_fmt='r-', camera_2_fmt='m-', camera_3_fmt='b-', camera_4_fmt='g-')

    plt.savefig("TESS_footprint.png")

    plt.show()
