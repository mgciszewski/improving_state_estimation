import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSS import *

def mirrored(maxval, inc):
    """
    Generates a neighbourhood of values mirrored around 0 with radius maxval and distance between values equal to inc

    Parameters
    ----------
    maxval : double
        maximum absolute value in an array
    inc : double
        the distance between subsequent values in an array

    Returns
    -------
    x: array of doubles
        an array containing zero and all multiples of inc whose absolute value is smaller than or equal to maxval.
    """
    # an array of all positive multiples of inc smaller than (not equal to) maxval
    x = np.arange(inc, maxval, inc)
    
    # if maxval is also a multiple of inc then add it to the array
    if x[-1] + inc == maxval:
        x = np.r_[x, maxval]
        
    # we add a mirror image of x and 0 to x
    x = np.r_[-x[::-1], 0, x]
        
    return x

def integral(jumps_f, jumps_g):
    """
    Calculates integral between two binary state sequences defined by their jumps

    The lengths of jumps_f and jumps_g are assumed to have the same parity

    Parameters
    ----------
    jumps_f : array of doubles
        an array of jumps of function f
    jumps_g : array of doubles
        an array of jumps of function g

    Returns
    -------
    integral: double
        the integral of |f-g|
    """
    # join and sort jump arrays of both functions
    jumps = np.sort(jumps_f + jumps_g)
    
    # calculates integral of |f-g| by summing differences between subsequent pairs of jumps in the joint sorted array
    integral = np.sum(np.array(jumps[1::2]) - np.array(jumps[::2]))
    
    return integral

def identify_jump_subseq(jumps, gam):
    """
    Returns array of arrays of jumps. 
    Jumps are put into one block (array) if the distance between subsequent jumps in the block is smaller than gam.

    Parameters
    ----------
    jumps : array of doubles
        an array of jumps
    gam : double
        lower bound on the lengths of the intervals (also double the weight of the jump)

    Returns
    -------
    res: an array of arrays of doubles
        a list of all blocks of jumps such that each block contains jumps that have at least one gamma neighbour in it or the block is a singleton
    """
    # res is initialized to a list containing one empty array; last refers to the last jump considered; at this moment it's None
    res, last = [[]], None
    
    # we iterate over all jumps
    for x in jumps:
        # if this is the first jump (last is None) or the previous jump was in gamma vicinity of 
        # current jump x then x is added to the most recent block in res
        if last is None or abs(last - x) < gam:
            res[-1].append(x)
        # otherwise a new block is created
        else:
            res.append([x])
        # we update last
        last = x
    
    return res

def flatten_list(l):
    """
    Flattens array of arrays; creates a single array with elements of all arrays in l.

    Parameters
    ----------
    l : array of arrays
        an array of arrays to flatten
        
    Returns
    -------
    an array consisting of elements from all arrays in l
    """
    return [item for sublist in l for item in sublist]

def set_size(width = 345.0, fraction=1, subplots=(1, 1)):
    """ 
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    # Dimensions of the figure as a tuple (in inches)
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def mystep(x,y, ax=None, **kwargs):
    """ 
    Plots discontinuous functions without vertical lines at jumps

    Parameters
    ----------
    x: list
        list of arguments
    y: list
        list of values
    ax: axis object
        Axis object on which the plot should appear

    Returns
    -------
    the plot of y against xW
    """
    # convert x and y into numpy array
    x = np.array(x)
    y = np.array(y)
    
    X = np.c_[x[:-1],x[1:],x[1:]]
    Y = np.c_[y[:-1],y[:-1],np.zeros_like(x[:-1])*np.nan]
    
    # if ax parameter is not given, use current figure
    if not ax: ax=plt.gca()
        
    return ax.plot(X.flatten(), Y.flatten(), **kwargs)

def neighbourhood_std(X):
    return pd.concat([X.loc[X.trial == trial, X.columns != 'trial'].rolling(130, min_periods = 1, center = True).std() for trial in X.trial.unique()])