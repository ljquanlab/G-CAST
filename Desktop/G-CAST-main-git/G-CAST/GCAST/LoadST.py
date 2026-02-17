import scanpy as sc
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import os
from tqdm import tqdm
import random
from pathlib import Path
import gc
from anndata import AnnData
import warnings
warnings.filterwarnings('ignore')

from .graph_func import graph_construction, combine_graph_dict


def load_truth(adata, file_path):
    df_meta = pd.read_csv(os.path.join(file_path, 'truth.txt'), sep='\t', header=None)
    df_meta_layer = df_meta[1]
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    return adata

def preprocess(adata,n_top_genes=3000):
    # adata.layers["count"] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)  ###
    # sc.pp.log1p(adata) ###
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3",  n_top_genes=n_top_genes)
    # print(adata)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    adata_X = PCA(n_components=200, random_state=2025).fit_transform(adata.X)
    adata.obsm["X_pca"] = adata_X
    return adata


def preprocess_batch(adata,n_top_genes=8000):
    # adata.layers["count"] = adata.X.to_numpy().toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)  ###
    # sc.pp.log1p(adata) ###
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    # print(adata)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    adata_X = PCA(n_components=200, random_state=2025).fit_transform(adata.X)
    adata.obsm["X_pca"] = adata_X
    return adata

def load_Histfeature( adata, file_path):
    scaler = StandardScaler()
    img_feat = np.load(os.path.join(file_path, 'embeddings.npy')).squeeze()
    img_feat = scaler.fit_transform(img_feat)
    adata_img = PCA(n_components=64, random_state=2025).fit_transform(img_feat)  ###
    adata.obsm["img"] = adata_img
    return adata


class Load10xST:
    def __init__(self, data_root,  sample_name, n_clusters, n_top_genes = 3000, negi= 12, tech="v10x", mode="single", label=True, Hist=True, random_seed=2025):
        self.data_root = data_root
        self.tech = tech
        self.sample_name = sample_name
        self.n_clusters = n_clusters
        self.negi = negi
        self.n_top_genes = n_top_genes
        self.mode = mode
        self.label = label
        self.Hist = Hist
        self.random_seed = random_seed

    def data(self):
        if self.mode == "single":
            file_path = os.path.join(self.data_root, self.sample_name)
            adata = sc.read_visium(path=file_path, )
            adata.var_names_make_unique()  # gene

            if self.label:
                adata = load_truth(adata, file_path)
            adata = preprocess(adata, self.n_top_genes)
            if self.Hist:
                adata = load_Histfeature(adata, file_path)
            graph_dict = graph_construction(adata, self.negi)


        elif self.mode == "batch":
            adata_list, graph_dict_lst = [], []
            for samplename in tqdm(self.sample_name):
                file_path = os.path.join(self.data_root, samplename)
                adata_tmp= sc.read_visium(file_path )
                adata_tmp.var_names_make_unique()
                adata_tmp.obs['batch_name'] = samplename

                if self.label:
                    adata_tmp = load_truth(adata_tmp, file_path)
                adata_tmp = preprocess_batch(adata_tmp, self.n_top_genes)
                if self.Hist:
                    adata_tmp = load_Histfeature(adata_tmp, file_path)
                graph_dict_tmp = graph_construction(adata_tmp, self.negi)

                adata_list.append(adata_tmp)
                graph_dict_lst.append(graph_dict_tmp)

            graph_dict = combine_graph_dict(*graph_dict_lst)
            adata = AnnData.concatenate(*adata_list)

            adata =  adata[:, adata.var['highly_variable'] == True]
            sc.pp.scale(adata)
            adata_X = PCA(n_components=200, random_state=2025).fit_transform(adata.X)
            adata.obsm['X_pca'] = adata_X
        return adata, graph_dict


class LoadCrossST:
    def __init__(self, data_root,  sample_name, file_name, n_clusters,tech_lst, subsample, Hist=True, negi=12,  n_top_genes=8000, random_seed=2025):
        self.data_root = data_root
        self.file_name = file_name
        self.tech_lst = tech_lst
        self.sample_name = sample_name
        self.n_top_genes = n_top_genes
        self.n_clusters = n_clusters
        self.negi = negi
        self.Hist = Hist
        self.subsample = subsample
        self.random_seed = random_seed

    def data(self):
        adata_list, graph_dict_lst = [], []
        for i in tqdm(range(len(self.tech_lst))):
            tech = self.tech_lst[i]
            samplename = self.sample_name[i]
            file_name = self.file_name[i]
            hist_label = self.Hist[i]
            subsample = self.subsample[i]


            if tech == "v10x":
                file_path = os.path.join(self.data_root, samplename)
                adata = sc.read_visium(file_path )
                adata.var_names_make_unique()
            else:
                file_path = os.path.join(self.data_root, samplename, file_name)
                adata = sc.read_h5ad(file_path)
                adata.var_names_make_unique()
                
            if subsample < 1:
                adata = sc.pp.subsample(adata, fraction=subsample, copy=True)
            elif subsample > 1:
                 adata = sc.pp.subsample(adata, n_obs=subsample, copy=True, random_state=2025)
                    
            adata.raw = adata.copy()
            adata.obs['batch_name'] =  file_name

            adata = preprocess_batch(adata, self.n_top_genes)

            if hist_label:
                hist_path = os.path.join(self.data_root, samplename)
                adata = load_Histfeature(adata, hist_path)
 
            graph_dict = graph_construction(adata, self.negi, emb=hist_label)

            adata_list.append(adata)
            graph_dict_lst.append(graph_dict)

        adata = AnnData.concatenate(*adata_list)
        graph_dict = combine_graph_dict(*graph_dict_lst)

        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)
        from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        return adata, graph_dict







