#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:38:16 2020

Based mostly on matplotlib tutorial

@author: mbattley
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

######################## Set font sizes #######################################
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
###############################################################################


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="",title ="", 
            xlabel ="", ylabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
#    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Set main and axis labels
#    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=0.5, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

periods = ["1.0", "2.0", "4.0", "6.0", "8.0", "10.0", "12.0", "14.0"]

radius_ratios = ["0.1", "0.075", "0.05", "0.04", "0.03"]

rotation_periods = ["0-1","1-2","2-3","3-4","4-5","5-6","6-7","7-8","8-9","9-10","10-11","11-12","12-13","13-14","14+"]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

sector_1 = np.array([[57, 50, 41, 29, 26],
                     [64, 50, 39, 29, 19],
                     [63, 48, 34, 28, 20],
                     [64, 47, 30, 20, 16],
                     [54, 45, 29, 19, 16],
                     [57, 42, 30, 23, 18],
                     [54, 37, 24, 14, 11],
                     [33, 28, 14,  6,  9]])

sector_1_percent = np.array([[0.77027027, 0.675675676, 0.554054054, 0.391891892, 0.351351351],
                             [0.86486486, 0.675675676, 0.527027027, 0.391891892, 0.256756757],
                             [0.85135135, 0.648648649, 0.459459459, 0.378378378, 0.27027027 ],
                             [0.86486486, 0.635135135, 0.405405405, 0.27027027,	0.216216216],
                             [0.72972973, 0.608108108, 0.391891892, 0.256756757,	0.216216216],
                             [0.77027027, 0.567567568, 0.405405405, 0.310810811,	0.243243243],
                             [0.72972973, 0.5,         0.324324324, 0.189189189,	0.148648649],
                             [0.44594594, 0.378378378, 0.189189189, 0.081081081,	0.121621622]])

sector_2_percent = np.array([[0.792207792,	0.805194805,	0.571428571,	0.454545455,	0.441558442],
                             [0.883116883,	0.701298701,	0.480519481,	0.402597403,	0.194805195],
                             [0.818181818,	0.662337662,	0.38961039,	0.337662338,	0.142857143],
                             [0.844155844,	0.649350649,	0.363636364,	0.324675325,	0.168831169],
                             [0.766233766,	0.571428571,	0.272727273,	0.233766234,	0.116883117],
                             [0.818181818,	0.558441558,	0.285714286,	0.220779221,	0.103896104],
                             [0.701298701,	0.532467532,	0.220779221,	0.155844156,	0.116883117],
                             [0.545454545,	0.337662338,	0.116883117,	0.155844156,	0.051948052]])

sector_3_percent = np.array([[0.68,        0.546666667, 0.4,         0.36,        0.306666667],
                             [0.8,         0.653333333, 0.493333333, 0.386666667, 0.32       ],
                             [0.853333333, 0.626666667, 0.466666667, 0.32,        0.266666667],
                             [0.8,         0.613333333, 0.4,         0.333333333, 0.186666667],
                             [0.8,         0.6        , 0.4,         0.293333333, 0.16       ],
                             [0.746666667, 0.586666667, 0.426666667, 0.28,        0.2        ],
                             [0.44,        0.346666667, 0.186666667, 0.2,         0.106666667],
                             [0.306666667, 0.213333333, 0.16,        0.146666667, 0.053333333]])

sector_4_percent = np.array([[0.658333333, 0.541666667, 0.391666667, 0.3,         0.225      ],
                             [0.783333333, 0.6,         0.383333333, 0.283333333, 0.15       ],
                             [0.816666667, 0.591666667, 0.391666667, 0.225,       0.133333333],
                             [0.733333333, 0.508333333, 0.308333333, 0.191666667, 0.091666667],
                             [0.683333333, 0.483333333, 0.283333333, 0.158333333, 0.05       ],
                             [0.65,        0.475,       0.241666667, 0.091666667, 0.033333333],
                             [0.558333333, 0.4,         0.208333333, 0.15,        0.091666667],
                             [0.35,        0.225,       0.075,       0.041666667, 0.025      ]])

sector_5_percent = np.array([[0.753623188, 0.702898551, 0.514492754, 0.420289855, 0.31884058 ],
                             [0.862318841, 0.746376812, 0.536231884, 0.442028986, 0.31884058 ],
                             [0.913043478, 0.775362319, 0.536231884, 0.420289855, 0.275362319],
                             [0.869565217, 0.746376812, 0.47826087,  0.369565217, 0.253623188],
                             [0.833333333, 0.717391304, 0.5,         0.384057971, 0.282608696],
                             [0.82608695,  0.717391304, 0.507246377, 0.347826087, 0.202898551],
                             [0.84057971,  0.644927536, 0.449275362, 0.275362319, 0.210144928],
                             [0.586956522, 0.463768116, 0.297101449, 0.173913043, 0.079710145]])

overall_percent = np.array([[0.727272727, 0.650826446, 0.481404959, 0.382231405, 0.318181818],
                            [0.83677686,  0.67768595,  0.481404959, 0.380165289, 0.247933884],
                            [0.855371901, 0.669421488, 0.454545455, 0.33677686,  0.216942149],
                            [0.820247934, 0.634297521, 0.394628099, 0.297520661, 0.183884298],
                            [0.76446281,  0.601239669, 0.378099174, 0.270661157, 0.169421488],
                            [0.760330579, 0.588842975, 0.378099174, 0.247933884, 0.150826446],
                            [0.669421488, 0.497933884, 0.29338843,  0.200413223, 0.140495868],
                            [0.45661157,  0.332644628, 0.175619835, 0.119834711,	0.064049587]])

rot_P_vs_depth = np.array([[0.596153846, 0.95,  0.95,  0.975, 0.894736842, 1,           0.944444444, 1,           0.842105263, 1,           1,     1,           0.944444444, 1,           0.841269841],
                           [0.317307692, 0.775, 0.75,  0.825, 0.789473684, 0.882352941, 0.944444444, 0.888888889, 0.789473684, 0.888888889, 0.625, 0.722222222, 0.722222222, 0.777777778, 0.682539683],
                           [0.125,       0.4,   0.375, 0.5,   0.631578947, 0.705882353, 0.722222222,	0.777777778, 0.736842105, 0.888888889, 0.5,   0.611111111, 0.555555556, 0.444444444, 0.547619048],
                           [0.067307692, 0.3,   0.1,   0.25,  0.315789474, 0.470588235, 0.555555556, 0.722222222, 0.578947368	, 0.666666667, 0.25,  0.5,         0.444444444, 0.333333333, 0.452380952],
                           [0.028846154,	 0.075, 0.05,  0.175, 0.157894737, 0.235294118, 0.388888889, 0.5,         0.473684211, 0.5,         0,     0.277777778, 0.388888889, 0.333333333, 0.341269841]])


overall_percent = overall_percent.transpose()

# Simple heat-map
#fig, ax = plt.subplots()
#im = ax.imshow(harvest)
#
## We want to show all ticks...
#ax.set_xticks(np.arange(len(farmers)))
#ax.set_yticks(np.arange(len(vegetables)))
## ... and label them with the respective list entries
#ax.set_xticklabels(farmers)
#ax.set_yticklabels(vegetables)
#
## Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")
#
## Loop over data dimensions and create text annotations.
#for i in range(len(vegetables)):
#    for j in range(len(farmers)):
#        text = ax.text(j, i, harvest[i, j],
#                       ha="center", va="center", color="w")
#
#ax.set_title("Harvest of local farmers (in tons/year)")
#fig.tight_layout()
#plt.show()

#################################################

# More complex heat-maps


# Find the min and max of all colors for use in setting the color scale.
#sector_list = [sector_1_percent,sector_2_percent,sector_3_percent,sector_4_percent,sector_5_percent]
#vmin = min(np.min(sector) for sector in sector_list)
#vmax = max(np.max(sector) for sector in sector_list)
#vmax = max(np.max(rot_P_vs_depth))
#vmin = min(np.min(rot_P_vs_depth))
#norm = colors.Normalize(vmin=vmin, vmax=vmax)

## Example
#fig_x, ax_x = plt.subplots()
#
#im, cbar = heatmap(harvest, vegetables, farmers, ax=ax_x,
#                   cmap="viridis", cbarlabel="harvest [t/year]")
#texts = annotate_heatmap(im, valfmt="{x:.1f}")
#
#fig_x.tight_layout()
#plt.show()

cbarlabel="Injected planets recovered (%)"
## Sector 1
#fig1, ax1 = plt.subplots()
#
#im, cbar = heatmap(sector_1_percent, periods, radius_ratios, ax=ax1,
#                   cmap="viridis", cbarlabel="Injected planets recovered (%)", 
#                   title ="Sector 1", xlabel="Injected R_P/R_*", ylabel="Injected Period (days)")
#im.set_norm(norm)
#cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#texts = annotate_heatmap(im, valfmt="{x:.2f}")
#
##fig1.tight_layout()
#plt.show()
#
## Sector 2
#fig2, ax2 = plt.subplots()
#
#im, cbar = heatmap(sector_2_percent, periods, radius_ratios, ax=ax2,
#                   cmap="viridis", cbarlabel="Injected planets recovered (%)", 
#                   title ="Sector 2", xlabel="Injected R_P/R_*", ylabel="Injected Period (days)")
#im.set_norm(norm)
#cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#texts = annotate_heatmap(im, valfmt="{x:.2f}")
#
##fig2.tight_layout()
#plt.show()
#
## Sector 3
#fig3, ax3 = plt.subplots()
#
#im, cbar = heatmap(sector_3_percent, periods, radius_ratios, ax=ax3,
#                   cmap="viridis", cbarlabel="Injected planets recovered (%)", 
#                   title ="Sector 3", xlabel="Injected R_P/R_*", ylabel="Injected Period (days)")
#im.set_norm(norm)
#cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#texts = annotate_heatmap(im, valfmt="{x:.2f}")
#
##fig3.tight_layout()
#plt.show()
#
## Sector 4
#fig4, ax4 = plt.subplots()
#
#im, cbar = heatmap(sector_4_percent, periods, radius_ratios, ax=ax4,
#                   cmap="viridis", cbarlabel="Injected planets recovered (%)", 
#                   title ="Sector 4", xlabel="Injected R_P/R_*", ylabel="Injected Period (days)")
#im.set_norm(norm)
#cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#texts = annotate_heatmap(im, valfmt="{x:.2f}")
#
##fig4.tight_layout()
#plt.show()
#
## Sector 5
#fig5, ax5 = plt.subplots()
#
#im, cbar = heatmap(sector_5_percent, periods, radius_ratios, ax=ax5,
#                   cmap="viridis", cbarlabel="Injected planets recovered (%)", 
#                   title ="Sector 5", xlabel="Injected R_P/R_*", ylabel="Injected Period (days)")
#im.set_norm(norm)
#cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#texts = annotate_heatmap(im, valfmt="{x:.2f}")
#
##fig5.tight_layout()
#plt.show()

# Overall
#fig6, ax6 = plt.subplots()
#
#im, cbar = heatmap(overall_percent, radius_ratios, periods, ax=ax6,
#                   cmap="viridis", cbarlabel="Injected planets recovered (%)", 
#                   title="Overall (S1-5)", xlabel="Injected Period (days)", ylabel="Injected R_P/R_*")
##im.set_norm(norm)
#texts = annotate_heatmap(im, valfmt="{x:.2f}")

#fig1.tight_layout()
#plt.show()

## Rotation Period vs Recovery Depth
rot_fig, rot_ax = plt.subplots()

im, cbar = heatmap(rot_P_vs_depth, radius_ratios, rotation_periods, ax=rot_ax,
                   cmap="viridis", cbarlabel="Injected planets recovered (%)", 
                   title ="", xlabel="Stellar Rotation Period (days)", ylabel="Injected R_P/R_*")
#im.set_norm(norm)
cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
texts = annotate_heatmap(im, valfmt="{x:.2f}")

#fig5.tight_layout()
plt.show()