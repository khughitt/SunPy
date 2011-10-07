from __future__ import absolute_import
"""
This module provides a set of colormaps specific to solar data (e.g. SDO/AIA 
color maps), functions for getting a colormap by name.
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from sunpy.cm import _cm

cmlist = {}
_aia_wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500]
_eit_wavelengths = [171, 195, 284, 304]

# AIA
for wl in _aia_wavelengths:
    key = "aia" + str(wl)
    cmlist[key] = colors.LinearSegmentedColormap(key, _cm.aia_color_table(wl))

# EIT
for wl in _eit_wavelengths:
    key = "eit" + str(wl)
    cmlist[key] = colors.LinearSegmentedColormap(key, _cm.eit_color_table(wl))
    
# RHESSI
cmlist['rhessi'] = cm.jet #pylint: disable=E1101

def get_cmap(name='aia94'):
    """Get a colormap instance."""
    if name in cmlist:
        return cmlist.get(name)
    else:
        raise ValueError("Colormap %s is not recognized" % name)

def show_colormaps():
    """Displays custom color maps supported in SunPy"""
    maps = sorted(cmlist)
    nmaps = len(maps) + 1
    
    a = np.linspace(0, 1, 256).reshape(1, -1) #pylint: disable=E1103
    a = np.vstack((a, a))
    
    fig = plt.figure(figsize=(5, 10))
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
    for i,name in enumerate(maps):
        ax = plt.subplot(nmaps, 1, i + 1)
        plt.axis("off")
        plt.imshow(a, aspect='auto', cmap=get_cmap(name), origin='lower')
        pos = list(ax.get_position().bounds)
        fig.text(pos[0] - 0.01, pos[1], name, fontsize=10, 
                 horizontalalignment='right')

    plt.show()
    
#
# Things to try:
#   use sub-sampling to speed up histogram generation
#
def adaptive_cmap(data, N=256, vmin=None, vmax=None, log=False, scale_factor=2):
    """Creates a custom log-scaled color map scaled to the specified image data.
    
    In order increase contrast in those pixel ranges which occur
    frequently a histogram is created for the log of the data and
    those values which occur most often are given unique colors.
    
    The method needs to be tested on other AIA, etc. data. For the sample
    image the color maps generated tend favor detail in the corona.
    
    So far little tweaking has been done though and there are probably
    things that could improved. Feel free to try!
       
    Parameters
    ----------
    data : numpy.ndarray
        Image data
    N    : int
        Number of colors to use for the color map
    vmin : int
        Minimum data clip value
    vmax : int
        Maximum data clip value
    log  : bool
        Whether or not to scale the data logarithmically during cmap creation
    scale_factor : float
        Scale factor to use when spreading out the histogram across the color
        map. Should be greater than 1, otherwise no optimization will be
        performed.
    
    Returns
    -------
    out : matplotlib.colors.LinearSegmentedColormap
        A grayscale color map normalized to the image data
        
    Examples
    --------
    >>> map = sunpy.Map(sunpy.AIA_171_IMAGE)
    >>> cmap = adaptive_cmap(map)
    >>> map.plot(cmap=cmap)    
    """
    
    # Applying clipping and log scaling if applicable
    if vmin is not None:
        data = data.clip(vmin)
    if vmax is not None:
        data = data.clip(None, vmax)

    if log:
        bins = _get_frequent_values(np.log(data.clip(1)), N, scale_factor)
        bins = np.exp(bins)
    else:
        bins = _get_frequent_values(data, N, scale_factor)

    # Scale from 0 to 1
    bins = bins / data.max() 
    
    # Create a matplotlib-formatted color dictionary
    cdict = _generate_cdict_for_indices(bins, N)
    
    return colors.LinearSegmentedColormap('automap', cdict, N)

def _generate_cdict_for_indices(indices, cmap_size):
    """Converts a list of indice values to an RGB color dictionary needed 
       to generate a linear segmented colormap
       
       See: http://matplotlib.sourceforge.net/api/colors_api.html
    """
    step = 1. / cmap_size
    cdict = {'red': [], 'green': [], 'blue': []}
    
    value = 0
    
    for i in indices:
        cmap_value = (i, value, value)
        cdict['red'].append(cmap_value)
        cdict['green'].append(cmap_value)
        cdict['blue'].append(cmap_value)
        value += step
        
    # cmap values must range from 0 to 1
    cdict['red'][0] = cdict['green'][0] = cdict['blue'][0] = (0, 0, 0)
    cdict['red'][-1] = cdict['green'][-1] = cdict['blue'][-1] = (1, 1, 1)
    
    # convert rgb lists to tuples
    cdict['red'] = tuple(cdict['red'])
    cdict['green'] = tuple(cdict['green'])
    cdict['blue'] = tuple(cdict['blue'])

    return cdict

def _get_frequent_values(data, N, scale_factor):
    """
    Gets a histogram of the image using a specified bin size that is greater
    than the number of color map indices (N). A sorted list of the N most
    frequent bins is then returned.
    """
    # Create a histogram
    hist, bins = np.histogram(data, bins=int(N * scale_factor))
    
    # Sort bins by frequency
    x = zip(hist, bins)
    x.sort(reverse=True)
    
    # Unzip and keep only the N most frequently occuring values
    hist, bins = zip(*x[:N])    
    
    # Return bins in ascending order
    return sorted(bins)

def test_equalize():
    '''Test'''
    dfile = cbook.get_sample_data('s1045.ima', asfileobj=False)
    
    im = np.fromstring(file(dfile, 'rb').read(), np.uint16).astype(float)
    im.shape = 256, 256

    #imshow(im, ColormapJet(256))
    #imshow(im, cmap=cm.jet)
    
    imvals = np.sort(im.flatten())
    lo = imvals[0]
    hi = imvals[-1]
    steps = (imvals[::len(imvals)/256] - lo) / (hi - lo)
    num_steps = float(len(steps))
    interps = [(s, idx/num_steps, idx/num_steps) for idx, s in enumerate(steps)]
    interps.append((1, 1, 1))
    cdict = {'red': interps,
             'green': interps,
             'blue': interps}
    histeq_cmap = colors.LinearSegmentedColormap('HistEq', cdict)
    pylab.figure()
    pylab.imshow(im, cmap=histeq_cmap)
    pylab.title('histeq')
    pylab.show()
