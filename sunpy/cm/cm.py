from __future__ import absolute_import
"""
This module provides a set of colormaps specific to solar data (e.g. SDO/AIA 
color maps), functions for getting a colormap by name.
"""
import sys
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from sunpy.util import util
from sunpy.cm import _cm

#pylint: disable=E1101

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
cmlist['rhessi'] = cm.jet

# Make accessible directly from module
for k, v in cmlist.items():
    setattr(sys.modules[__name__], k, v)

def get_cmap(name):
    """Get a colormap instance."""
    if name in cmlist:
        return cmlist.get(name)
    else:
        raise ValueError("Colormap %s is not recognized" % name)
    
def get_adaptive_cmap(data, cmap=None, N=256, vmin=None, vmax=None, 
                      log=False, scale_factor=1.5):
    """Returns an optimized colormap for the specified data.
    
    The image data (histogram) is used to construct a custom colormap based on
    an existing color map. The function works by using the image's histogram
    to determine where the most frequently occuring pixel values lie. Colormap
    indices are then weighted more towards those values.
       
    Parameters
    ----------
    data : numpy.ndarray
        Image data
    cmap : string
        Name of the color map to use as a base
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
    >>> aia = sunpy.Map(sunpy.AIA_171_IMAGE)
    >>> cmap = adaptive_cmap(aia, sunpy.cm.aia171)
    >>> map.plot(cmap=cmap)
    """
    # Applying clipping and log scaling if applicable
    if vmin is not None:
        data = data.clip(min=vmin)
    if vmax is not None:
        data = data.clip(max=vmax)

    if log:
        bins = _get_frequent_values(np.log(data.clip(1)), N, scale_factor)
        bins = np.exp(bins)
    else:
        bins = _get_frequent_values(data, N, scale_factor)

    # Scale from 0 to 1
    bins = bins / data.max()
    
    # Default to a grayscale cmap
    if cmap is None:
        cdict = _generate_cdict_for_indices(bins, N)
        return colors.LinearSegmentedColormap("gray_adaptive", cdict, N)
    
    # Or use specified cmap as a base
    else:
        return _adjust_cmap_indices(cmap, bins, N)

def _adjust_cmap_indices(cmap, indices, N):
    """Creates a copy of a color map with the same interpolation values but
    new indices.
    
    Unsupported:
        - colormaps with > N indices
        - colormaps which use functions instead of tuples for rgb indices
    """
    name = cmap.name
    new_cmap = {"red": [], "green": [], "blue": []}
    
    # Get original segment data and interpolate if needed
    cdict = _interpolate_cmap_indices(cmap._segmentdata, N) #pylint: disable=W0212

    i = 0
    for index in indices:
        rval = (index, cdict['red'][i][1], cdict['red'][i][2])
        new_cmap['red'].append(rval)
        
        gval = (index, cdict['green'][i][1], cdict['green'][i][2])
        new_cmap['green'].append(gval)
        
        bval = (index, cdict['blue'][i][1], cdict['blue'][i][2])
        new_cmap['blue'].append(bval)
        
        i += 1
        
    # cmap values must range from 0 to 1
    new_cmap['red'][0] = new_cmap['green'][0] = new_cmap['blue'][0] = (0, 0, 0)
    new_cmap['red'][-1] = new_cmap['green'][-1] = new_cmap['blue'][-1] = (1, 1, 1)
        
    return colors.LinearSegmentedColormap(name + "_adaptive", new_cmap, N)

def _generate_cdict_for_indices(indices, N):
    """Converts a list of indice values to an RGB color dictionary needed 
       to generate a linear segmented colormap
       
       See: http://matplotlib.sourceforge.net/api/colors_api.html
    """
    x0 = np.linspace(0, 1, N)
    cmap_values = zip(indices, x0, x0)
    
    cdict = {'red': cmap_values, 'green': cmap_values, 'blue': cmap_values}    
        
    # cmap values must range from 0 to 1
    cdict['red'][0] = cdict['green'][0] = cdict['blue'][0] = (0, 0, 0)
    cdict['red'][-1] = cdict['green'][-1] = cdict['blue'][-1] = (1, 1, 1)
    
    # convert rgb lists to tuples
    cdict['red'] = tuple(cdict['red'])
    cdict['green'] = tuple(cdict['green'])
    cdict['blue'] = tuple(cdict['blue'])

    return cdict

def _interpolate_cmap_indices(cdict, N):
    """Expands the input indices into N values"""
    new_cdict = {"red": [], "green": [], "blue": []}
   
    for color in cdict.keys():
        if len(cdict[color]) != N:
            # Get the original values
            x, y0, y1 = zip(*cdict[color])
            
            # Interpolate up to N indices
            new_cdict[color] = tuple(zip(
                                         tuple(util.interpolate(x, N)),
                                         tuple(util.interpolate(y0, N)),
                                         tuple(util.interpolate(y1, N))
                                         ))
        else:
            new_cdict[color] = cdict[color]
            
    return new_cdict

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
    return np.array(sorted(bins))

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
    
if __name__ == "__main__":
    import sunpy
    from matplotlib import cm
    aia = sunpy.Map(sunpy.AIA_171_IMAGE)
    #cmap = sunpy.cm.get_adaptive_cmap(aia, sunpy.cm.aia171, vmin=0, vmax=2000, scale_factor=1.5, log=True)
    cmap = sunpy.cm.get_adaptive_cmap(aia, cm.jet)
