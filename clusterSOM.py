"""-----------------------------------------------------------------------------
Class clusterSOM - summarizes functionalities and attributes of kmeans-clustered
                    Self-Organizing-Maps
author - Sophia Köhne (sophia.koehne@rub.de)
-----------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from mpl_toolkits import axes_grid1
import matplotlib as mpl
from utility_functions import *
from os import path

class clusterSOM:
    '''contains k-means clustered SOM and data'''
    def __init__(self, init_dir):
        '''initialization, loading the relevant attributes from pickled objects'''
        #the trained weights of the SOM, size: (ysize*xsize, n_features)
        with open(init_dir+'/SOM_weights.p', 'rb') as f:
            self.SOM_weights = pickle.load(f)

        #k-means labels for all trained weights of the SOM, size: (ysize*xsize)
        with open(init_dir+'/SOM_labels.p', 'rb') as f:
            self.SOM_labels = pickle.load(f)

        #k-means instance whith which the SOM has been clustered
        with open(init_dir+'/SOM_kmeans.p', 'rb') as f:
            self.SOM_kmeans = pickle.load(f)

        #index integer to the BMU in SOM_weights for each data sample, shape (n_samples, 2) (first row is 0:n_samples)
        with open(init_dir + '/BMU_matches.p', 'rb') as f:
            self.BMU_matches = pickle.load(f)

        #label for each data sample corresponding to label of BMU in SOM_labels
        with open(init_dir+'/data_labels.p', 'rb') as f:
            self.data_labels = pickle.load(f)

        #dict containing parameters of the SOM and training
        with open(init_dir+'/params_SOM.p', 'rb') as f:
            self.params_SOM = pickle.load(f)

        #dict containing info about the simulation plane
        with open(init_dir+'/params_2D.p', 'rb') as f:
            self.params_2D = pickle.load(f)

        #scaler with which data has been scaled
        with open(init_dir + '/scaler.p', 'rb') as f:
            self.scaler = pickle.load(f)

        if path.isfile(init_dir+'/contour.p'):
            #contour with values -1, 0, 1 for the plotting of the boundaries of the clusters in the SOM
            with open(init_dir + '/contour.p', 'rb') as f:
                self.contour = pickle.load(f)
        else:
            self.contour = None
        self.features = self.params_2D['features']


    def plot_umatrix(self, ax, contour = False, title = ''):
        '''plots the unified distance matrix of the SOM'''
        reshaped = np.reshape(self.SOM_weights,(self.params_SOM['ysize'],self.params_SOM['xsize'],len(self.params_2D['features'])))
        if (contour == True) and (self.contour is not None):
            ax.contour(np.reshape(self.contour, (self.params_SOM['ysize'], self.params_SOM['xsize'])), 1, colors = 'k', linewidths = 1.2, nchunk = len(np.unique(self.SOM_labels)))
        pic = ax.pcolor(distance_map(reshaped), cmap='bone_r', alpha = 0.8)  # plotting the distance map as background
        ax.set_xticks(np.arange(0, self.params_SOM['xsize'], 10), [str(tick) for tick in np.arange(0, self.params_SOM['xsize'], 10)])
        ax.set_yticks(np.arange(0, self.params_SOM['ysize'], 10), [str(tick) for tick in np.arange(0, self.params_SOM['ysize'], 10)])
        ax.set_title(title)
        plt.colorbar(pic, ax = ax)
        return ax

    def plot_SOM_kmeans(self, ax, title = ''):
        '''plots the k-means cluster labels of the SOM nodes in different colors '''
        colors =  ['C0', 'C1', 'C2', 'C3', 'C4', 'C6', 'C7', 'C8']
        cmap = mpl.colors.LinearSegmentedColormap.from_list('cluster',colors[0:5])
        mpl.cm.register_cmap(name='cluster', cmap=cmap)
        ax = visualize_SOMclusters_simple(ax, np.reshape(self.SOM_labels, (self.params_SOM['ysize'],self.params_SOM['xsize'])), cluster = len(np.unique(self.SOM_labels)))#, cmap = cmap)
        ax.set_title(title)
        return ax

    def plot_2D_kmeans(self, ax, title = ''):
        '''plots the data samples in the color of their BMUs k-means cluster labels'''
        colors =  ['C0', 'C1', 'C2', 'C3', 'C4', 'C6', 'C7', 'C8']
        cmap = mpl.colors.LinearSegmentedColormap.from_list('cluster',colors[0:5])
        mpl.cm.register_cmap(name='cluster', cmap=cmap)
        heatmap2d(ax, np.reshape(self.data_labels, (self.params_2D['ysize'], self.params_2D['xsize'])),title, ymin = self.params_2D['ymin'], ymax= self.params_2D['ymax'], cluster = len(np.unique(self.SOM_labels)))#, colmap=cmap)
        return ax

    def feature_map(self, ax, feature_key = 'Bx', contour = False, title  = ''):
        '''plots the distribution of the weight values of a specific feature in the SOM'''
        rescaled_weights = np.reshape(self.scaler.inverse_transform(self.SOM_weights), (self.params_SOM['ysize'],self.params_SOM['xsize'],len(self.params_2D['features'])))
        feature_weights = rescaled_weights[:,:,self.params_2D['features'].index(feature_key)]
        if (contour == True) and (self.contour is not None):
            ax.contour(np.reshape(self.contour, (self.params_SOM['ysize'], self.params_SOM['xsize'])), 1, colors = 'k', linewidths = 1.2, nchunk = len(np.unique(self.SOM_labels)))

        vmin, vmax, vcenter = get_normvals(feature_weights)
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        im = ax.pcolor(feature_weights, cmap = 'seismic' ,norm = norm , alpha = 0.7)

        cb = add_colorbar(im, spacing = 'uniform')

        ax.set_title(title)#, fontsize = 20)
        ax.set_xticks(np.arange(0, self.params_SOM['xsize'], 10), [str(tick) for tick in np.arange(0, self.params_SOM['xsize'], 10)])#, fontsize = 22)
        ax.set_yticks(np.arange(0, self.params_SOM['ysize'], 10), [str(tick) for tick in np.arange(0, self.params_SOM['ysize'], 10)])#, fontsize = 22)
        return ax

    def plot_handpicked_region(self, tpls, title = ''):
        '''plots the k-means labels of the SOM nodes in different colors + the handpicked region in black'''
        #colormap
        colors =  ['C0', 'C1', 'C2', 'C3', 'C4', 'k', 'C9', 'C9']
        cmap6 = mpl.colors.LinearSegmentedColormap.from_list('cluster',colors[0:6])
        mpl.cm.register_cmap(name='cluster', cmap=cmap6)

        tpls__ = [(sub[1], sub[0]) for sub in tpls]
        blob_matches_BMUs = []
        blob_matches_ = []

        for i, BMU in enumerate(self.BMU_matches): #iterate all indices of BMUs
            #get the BMU index in tuple form in the map
            BMUtpl = np.unravel_index(int(BMU[1]), (self.params_SOM['ysize'],self.params_SOM['xsize']))
            if BMUtpl in tpls__: #check whether tuple is in the handpicked region
                blob_matches_BMUs.append(BMUtpl) #append the tuple
                blob_matches_.append(i)

        blob_matches_BMUs_ravel = np.zeros(len(tpls__))
        for i in range(0,len(tpls__)):
            #ravel the indices back up
            blob_matches_BMUs_ravel[i] = np.ravel_multi_index(tpls__[i],(self.params_SOM['ysize'],self.params_SOM['xsize']))

        hp_lbls_ = self.SOM_labels.copy()
        hp_lbls_[(blob_matches_BMUs_ravel).astype(int)] = 5 #make the handpicked region artifical 6th cluster

        data_lbls = BMU_datalbls(self.BMU_matches, hp_lbls_)

        fig = plt.figure(figsize = (10,10))
        #shift the upper panel to the middle above the lower
        gs = fig.add_gridspec(2,10)
        f_ax1 = fig.add_subplot(gs[0, 2:8])
        f_ax2 = fig.add_subplot(gs[1, :])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)

        f_ax1 = visualize_SOMclusters_simple(f_ax1, np.reshape(hp_lbls_, (self.params_SOM['ysize'],self.params_SOM['xsize'])))
        f_ax1.set_title('a)')

        #handpick eval
        f_ax2 = heatmap2d(f_ax2,np.reshape(data_lbls,(self.params_2D['ysize'],self.params_2D['xsize'])),'', ymin = self.params_2D['ymin'], ymax = self.params_2D['ymax'], cluster=6, filename = '')
        f_ax2.set_title('b)')
        return (fig)

    def SOM_matching(self, SOM_tomatch):
        '''calculates the matching factor R of two clusterSOMs (clustered SOMs)'''
        clusters = np.unique(self.data_labels)
        matches = np.zeros(len(clusters))
        counts = np.zeros(len(clusters))

        ref = np.reshape(self.data_labels,-1)
        to_match = np.reshape(SOM_tomatch.data_labels,-1)

        for clust in clusters:
            if clust == 0: #exclude the inflow region
                continue

            ind_ref = np.where(ref == int(clust))
            ind_match = np.where(to_match == int(clust))
            print('finding matches for cluster number ', clust)
            count = 0
            for sample in ind_ref[0]:
                if np.where(ind_match == sample)[1].size > 0:
                    count += 1
            matches[int(clust)] = count/len(ind_ref[0])
            counts[int(clust)] = count
        R = np.sum(counts) / len(np.where(ref != 0)[0])
        print('R = ', np.round(R,2))
        return (R, matches, counts)
