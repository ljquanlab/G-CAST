import warnings

from esm.utils.constants.esm3 import data_root

warnings.filterwarnings('ignore')

import scanpy as sc
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os

from anndata import AnnData

import GCAST

device = "cuda" if torch.cuda.is_available() else "cpu"


def GCAST_Runner(path,
                n_clusters,
                sample_name="sample",
                label=False,
                negi=12,
                mode="Single",
                cluster_method="KMeans",
                Histology=True,
                random_seed=2025,
                 ):
    """

    Parameters
    ----------
    path
    n_clusters
    sample_name
    tech
    label
    negi
    Mode: Single;batch (ST);
    cluster_method: KMeans;KMedoids;Mclust;louvain;leiden
    Histology
    random_seed
    n_top_genes
    output

    Returns
    -------

    """
    GCAST.set_seed(random_seed)

    if mode == "single":
       adata, graph_dict = GCAST.Load10xST(data_root=path, sample_name=sample_name,
                                           n_clusters=n_clusters,mode=mode, negi=negi,
                                           label=label, Hist=Histology,
                                           ).data()
    elif mode == "batch":
        adata, graph_dict = GCAST.Load10xST(data_root=path, sample_name=sample_name,
                                            n_clusters=n_clusters, mode=mode, negi=negi,
                                            label = label, Hist = Histology,
                                            ).data()
    # print(adata)

    print("Training GCAST model...")
    net = GCAST.GCAST(adata.obsm["X_pca"], graph_dict, device=device, model_name=sample_name)


    net.train_with_dec()

    net.load_model()
    feat, _, _ = net.eval_model()
    adata.obsm["emb"] = feat

    if cluster_method == "KMeans":
        GCAST.KMeans_Cluster(adata, n_clusters=n_clusters, use_rep="emb", key_added=f"{cluster_method}", random_seed=random_seed)
    elif cluster_method == "KMedoids":
        GCAST.KMedoids_Cluster(adata, n_clusters=n_clusters, use_rep="emb", key_added=f"{cluster_method}", random_seed=random_seed)
    elif cluster_method == "mclust":
        GCAST.mclust_R(adata, n_clusters=n_clusters, use_rep="emb", key_added=f"{cluster_method}", random_seed=random_seed)
    elif cluster_method == "louvain":
        GCAST.louvain(adata, n_clusters=n_clusters, use_rep="emb", key_added=f"{cluster_method}", random_seed=random_seed)
    elif cluster_method == "leiden":
        GCAST.leiden(adata, n_clusters=n_clusters, use_rep="emb", key_added=f"{cluster_method}", random_seed=random_seed)

    GCAST.evaluate_cluster(adata, mode=label, GTkey="ground_truth", method_name=f"{cluster_method}")
    GCAST.plot_spatial(adata, color=f"{cluster_method}", title=f"{sample_name}_{cluster_method}", save=False, show=True)
