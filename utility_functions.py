from mpl_toolkits import axes_grid1
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_scatter_density
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, SymLogNorm, Normalize
import pandas as pd
import seaborn as sns

#classification colormap
colors =  ['C0', 'C1', 'C2', 'C3', 'C4', 'C6', 'C7', 'C8']
cmap = mpl.colors.LinearSegmentedColormap.from_list('test',colors[0:5])
mpl.cm.register_cmap(name='cluster', cmap=cmap)

#make negative contours solid lines
mpl.rcParams.update({"contour.negative_linestyle" : 'solid'})

#colormap for the brazilian plot
cm = plt.cm.viridis.copy()
viridis_cust = cm
viridis_cust.set_under(alpha = 0)

'''Brazilian Parameters'''
beta_plot = np.logspace(-4, 4, num = 1000)

#electrons
S1= 1.29
alpha1=0.97

S2= 1.32
alpha2=0.61

S3= 1.36
alpha3=0.47

#whistler waves
a1= 0.36
b1=0.55
a2= 1.0
b2=0.49

#calculate the thresholds
y_plot1 = 1-S1/(beta_plot**alpha1)
y_plot2 = 1-S2/(beta_plot**alpha2)
y_plot3 = 1-S3/(beta_plot**alpha3)
y_plot_whistler1 = 1 + a1/(beta_plot**b1)
y_plot_whistler2 = 1 + a2/(beta_plot**b2)

'''#############################################################################'''

def BMU_datalbls(BMUmatches, labels):
    '''takes the BMUmatches of the samples and the k-means labels of the SOM to produce
    the k-means labels of the data'''
    data_lbls = np.zeros(len(BMUmatches))
    for i in range(0,len(BMUmatches)):
        data_lbls[i] = labels[int(BMUmatches[i][1])]
    return data_lbls

def stability_thresh_density(fig, pos, beta, tratio, cluster_num = None, label = ''):
    '''calculate the scatter density with mpl_scatter_density and plot the brazilian plots'''
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C8', 'C9', 'saddlebrown']

    if cluster_num == None:
        color = 'b'
        cmap = 'Greys'
    else:
        color = colors[cluster_num]
        cmap = 'plasma'

    ax = fig.add_subplot(pos[0], pos[1], pos[2], projection='scatter_density')
    ax.grid(False)
    density = ax.scatter_density(beta,tratio, cmap= viridis_cust, vmin = 0.55, vmax = np.nanmax)#, color = color)
    fig.colorbar(density)

    #thresholds
    ax.plot(beta_plot, y_plot1, c = 'k', linestyle = 'solid')
    ax.plot(beta_plot, y_plot2, c = 'k' , linestyle = 'dashed')
    ax.plot(beta_plot, y_plot3, c = 'k' , linestyle = 'dotted')
    ax.plot(beta_plot, y_plot_whistler1, c = 'k', linestyle = 'solid')
    ax.plot(beta_plot, y_plot_whistler2, c = 'k' , linestyle = 'dashed')

    if cluster_num == 7:
        cluster_num = 'None'
    ax.set_xlabel(r'$\beta_{\parallel, e}$')
    ax.set_ylabel(r'$T_{\perp,e}/ T_{\parallel,e}$')
    ax.set_title(label)
    #scale plot
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlim(1e-3, 1e2)
    return ax

def slicedata(array,res = 2128/200, xmin = 0,xmax = 200, ymin = 0, ymax = 200):
    """slices the data array at the positions defined in xmin, xmax, ymin, ymax
    returns reduced array
        array : numpy array of size (y_samples, x_samples)
            to be sliced
        res : float
            n_samples/boxlength
        xmin, ymin : integers or floats
            x and y position of boxlength were lower limit of cut array should be
        xmax, ymax : integers or floats
            x and y position of boxlength were upper limit of cut array should be
    """
    slicedarr = array
    if ymin != 0 or ymax != 200:
        slicedarr=slicedarr[int(res*ymin):int(ymax*res),:]
    if xmin != 0 or xmax != 200:
        slicedarr = slicedarr[:,int(res*xmin):int(res*xmax)]
    return(slicedarr)

def visualize_SOMclusters_simple(ax, labels, cluster = 6, cmap = None):
    '''plots the given SOM labels in corresponding colors'''
    if cmap:
        colmap = cmap
    else:
        colmap = mpl.cm.get_cmap('cluster', cluster)
    im = ax.imshow(labels, cmap = colmap, origin ='lower', aspect = 'auto')

    labels = labels.astype(int)
    ax.set_xticks(np.arange(0, labels.shape[1], 10),[str(tick) for tick in np.arange(0, labels.shape[1], 10)])#, fontsize = 25)
    ax.set_yticks(np.arange(0, labels.shape[0], 10),[str(tick) for tick in np.arange(0, labels.shape[0], 10)])#, fontsize = 25)

    mini = np.min(labels)
    maxi = np.max(labels)
    num = len(np.unique(labels))
    bounds = np.linspace(mini, maxi, num +1 )
    add = (bounds[1]-bounds[2])/2
    yticks = bounds+add
    cb = add_colorbar(im, ticks = yticks[1:] )
    cb.ax.set_yticklabels(np.arange(0,cluster))
    return ax

def distance_map(weights):
    '''calculates the distance map for the unified distance matrix'''
    um = np.zeros((weights.shape[0],weights.shape[1],8))  # 2 spots more for hexagonal topology

    ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
    jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2

    for x in range(weights.shape[0]):
        for y in range(weights.shape[1]):
            w_2 = weights[x, y]
            e = y % 2 == 0   # only used on hexagonal topology
            for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                if (x+i >= 0 and x+i < weights.shape[0] and
                        y+j >= 0 and y+j < weights.shape[1]):
                    w_1 = weights[x+i, y+j]
                    um[x, y, k] = fast_norm(w_2-w_1)

    um = um.sum(axis=2)
    return um/um.max()

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return np.sqrt(np.dot(x, x.T))

def heatmap2d(ax, arr : np.ndarray,ttl,xmin = 0,xmax = 200, ymin = 0, ymax = 200,norm = None, colmap=None,dir="Clusters", cluster=0, filename = '', power =False, f=None):
    '''plots the arr in the 2D plane'''
    xsize=arr[0,:].size
    ysize=arr[:,0].size

    ax.set_ylim(0,ysize)
    ax.set_xlim(0,xsize)

    xticks = np.linspace(0,xsize-1, 11)
    yticks = np.linspace(0,ysize-1,int((ymax-ymin)/20)+1)
    xlabels = np.linspace(xmin,xmax,11).astype(int)
    ylabels = np.linspace(ymin,ymax,int((ymax-ymin)/20)+1).astype(int)

    ax.set_xticks(xticks, xlabels)#, fontsize = 25)
    ax.set_yticks(yticks, ylabels)#, fontsize = 25)

    if cluster != 0:# and colmap not None:
        if colmap:
            cmap = colmap
        else:
            cmap = mpl.cm.get_cmap('cluster', cluster)
        colmap = cmap

    im = ax.imshow(arr, cmap=colmap,origin='lower',interpolation='none',  norm=norm)#norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter))# norm = TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0)) #vmin = -0.35, vmax = 0)# norm=SymLogNorm(linthresh=0.01, vmin=-0.07, vmax=0.07) )
    if cluster != 0:
        mini = np.min(arr)
        maxi = np.max(arr)
        num = len(np.unique(arr))
        bounds = np.linspace(mini, maxi, num +1 )
        add = (bounds[1]-bounds[2])/2
        yticks = bounds+add
        cb = add_colorbar(im, ticks = yticks[1:] )
        cb.ax.set_yticklabels(np.arange(0,cluster))
    else:
        cb = add_colorbar(im, spacing = 'proportional')

    if power:
        cb.formatter.set_powerlimits((0, 0))

    if f is not None:
        ax.contour(f,25,colors = 'grey', linestyles="solid")
    ax.set_title(ttl) #, size = 35)
    ax.set_xlabel("$x/d_i$") #, fontsize = 30)
    ax.set_ylabel("$y/d_i$") #, fontsize = 30)
    return (ax)

def get_normvals(arr):
    '''get the norm values for plotting norm for given array arr'''
    if arr.max() <= 0:
        vmax = None
        vcenter = 0
        vmin = arr.min()
    elif arr.min () >= 0:
        vmin = None
        vmax = arr.max()
        vcenter = 0
    else:
        vmax = arr.max()
        vmin = arr.min()
        vcenter = 0

    return(vmin, vmax, vcenter)

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
