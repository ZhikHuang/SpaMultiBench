import pandas as pd
import seaborn as sns
import anndata as ad
import numpy as np
import squidpy as sq
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix
from scipy.stats import ranksums
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

# Moran's I calculation for clusters
def weighted_moran_I(adata,method):
    n_clusters = adata.obs[method].unique()
    total_samples = adata.X.shape[0]

    moran_indices = []
    cluster_sizes = []

    for cluster in n_clusters:
        column_name = f'cluster_{cluster}'
        adata.obs[column_name] = np.where(adata.obs[method] == cluster, 1, 0)
        
  
    sq.gr.spatial_neighbors(adata)

    for cluster in n_clusters:
        column_name = f'cluster_{cluster}'
        
        
        cluster_size = adata.obs[column_name].sum()
        cluster_sizes.append(cluster_size)
        
        
        sq.gr.spatial_autocorr(adata, mode='moran', genes=[column_name], attr='obs')
        moran_index = adata.uns['moranI']['I'][0]
        moran_indices.append(moran_index)

   
    moran_indices = np.array(moran_indices)
    cluster_sizes = np.array(cluster_sizes)

    weighted_moran = np.sum((cluster_sizes / total_samples) * moran_indices)

    return weighted_moran

# CHAOS calculation for clusters
def _compute_CHAOS(clusterlabel, location):

        clusterlabel = np.array(clusterlabel)
        location = np.array(location)
        matched_location = StandardScaler().fit_transform(location)

        clusterlabel_unique = np.unique(clusterlabel)
        dist_val = np.zeros(len(clusterlabel_unique))
        count = 0
        for k in clusterlabel_unique:
            location_cluster = matched_location[clusterlabel==k,:]
            if len(location_cluster)<=2:
                continue
            n_location_cluster = len(location_cluster)
            results = [fx_1NN(i,location_cluster) for i in range(n_location_cluster)]
            dist_val[count] = np.sum(results)
            count = count + 1

        return np.sum(dist_val)/len(clusterlabel)

def fx_1NN(i,location_in):
        location_in = np.array(location_in)
        dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
        dist_array[i] = np.inf
        return np.min(dist_array)
    

def fx_kNN(i,location_in,k,cluster_in):

        location_in = np.array(location_in)
        cluster_in = np.array(cluster_in)


        dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
        dist_array[i] = np.inf
        ind = np.argsort(dist_array)[:k]
        cluster_use = np.array(cluster_in)
        if np.sum(cluster_use[ind]!=cluster_in[i])>(k/2):
            return 1
        else:
            return 0
        
def _compute_PAS(clusterlabel,location):
        
        clusterlabel = np.array(clusterlabel)
        location = np.array(location)
        matched_location = location
        results = [fx_kNN(i,matched_location,k=10,cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
        return np.sum(results)/len(clusterlabel)

def compute_CHAOS(adata,pred_key,spatial_key='spatial'):
        return _compute_CHAOS(adata.obs[pred_key],adata.obsm[spatial_key])

# PAS calculation for clusters
def compute_PAS(adata,pred_key,spatial_key='spatial'):
        return _compute_PAS(adata.obs[pred_key],adata.obsm[spatial_key])



# adjusted silhouette width calculation for clusters, silhouette score ranges from -1 to 1, rescale to 0 to 1
def compute_ASW(adata,pred_key,spatial_key='spatial'):
        d = squareform(pdist(adata.obsm[spatial_key]))
        score_rescaled = (silhouette_score(X=d,labels=adata.obs[pred_key],metric='precomputed') + 1) / 2
        return score_rescaled

# purity score calculation for clusters
def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)

# cluster entropy calculation for clusters
def cluster_entropy(y_true, y_pred, base=np.e):
    cm = contingency_matrix(y_true, y_pred)
    cluster_sizes = np.sum(cm, axis=0)            
    entropies = []
    for j in range(cm.shape[1]):
        p_ij = cm[:, j] / cluster_sizes[j]   
      
        nz = p_ij > 0
        h_j = -np.sum(p_ij[nz] * np.log(p_ij[nz]) / np.log(base))
        entropies.append(h_j)
    return np.sum((cluster_sizes / np.sum(cluster_sizes)) * np.array(entropies))

# weighted F1 score calculation for clusters
def caculate_weighted_F1_score(adata,method): 
    from sklearn.metrics import f1_score
    df=adata.obs.copy()
    f1_scores = {}
    weights = {}
    for label in df['Ground Truth'].unique():
                df['Ground Truth_bin'] = (df['Ground Truth'] == label).astype(int)
                
               
                cluster_ratio = df.groupby(f'{method}')['Ground Truth_bin'].mean()
            
                cluster_label = (cluster_ratio > 0.5).astype(int)
            
                df['cluster_pred'] = df[f'{method}'].map(cluster_label)

                f1 = f1_score(df['Ground Truth_bin'], df['cluster_pred'])

                f1_scores[label] = f1

                weights[label] = (df['Ground Truth'] == label).mean()

    weighted_f1= sum(f1_scores[label] * weights[label] for label in f1_scores)

    return weighted_f1

