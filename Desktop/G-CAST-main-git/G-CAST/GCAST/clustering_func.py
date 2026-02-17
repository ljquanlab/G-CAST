import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
from scipy.spatial.distance import cdist
import anndata
from  anndata import AnnData
import harmonypy as hm
from typing import Tuple, Optional


def KMeans_Cluster(adata, n_clusters, use_rep="emb", key_added="KMeans", random_seed=2025):
    labels = KMeans(n_clusters=n_clusters, random_state=random_seed).fit_predict(adata.obsm[use_rep])
    adata.obs[key_added] = labels
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

def KMedoids_Cluster(adata, n_clusters, use_rep="emb", key_added="KMedoids", random_seed=2025):
    labels = KMedoids(n_clusters=n_clusters, random_state=random_seed).fit_predict(adata.obsm[use_rep])
    adata.obs[key_added] = labels
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = cdist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type




def res_search_fixed_clus_leiden(adata, n_clusters, max_res=3, min_res=0.1, random_seed=2025, tol=0.1, max_iters=50):
    low = min_res
    high = max_res
    best_res = min_res
    best_diff = float('inf')

    for _ in range(max_iters):
        mid = (low + high) / 2
        sc.tl.leiden(adata, resolution=mid, random_state=random_seed, key_added='leiden_temp')
        n_found = len(adata.obs['leiden_temp'].unique())

        diff = abs(n_found - n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_res = mid

        if n_found < n_clusters:
            low = mid
        elif n_found > n_clusters:
            high = mid
        else:
            best_res = mid
            break

        if best_diff <= tol:
            break
    return best_res


def leiden(adata, n_clusters, key_added='', use_rep="emb", random_seed=2025):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, random_seed=random_seed)
    sc.tl.leiden(adata, resolution=res, random_state=random_seed)
    adata.obs[key_added] = adata.obs['leiden'].astype('int').astype('category')
    n_found = len(adata.obs[key_added].unique())
    if n_found != n_clusters:
        print(f"Warning: Found {n_found} clusters instead of {n_clusters}")
    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, max_res=3, min_res=0.1, random_seed=2025, tol=0.1, max_iters=50):
    low = min_res
    high = max_res
    best_res = min_res
    best_diff = float('inf')

    for _ in range(max_iters):
        mid = (low + high) / 2
        sc.tl.louvain(adata, resolution=mid, random_state=random_seed, key_added='louvain_temp')
        n_found = len(adata.obs['louvain_temp'].unique())

        diff = abs(n_found - n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_res = mid

        if n_found < n_clusters:
            low = mid
        elif n_found > n_clusters:
            high = mid
        else:
            best_res = mid
            break

        if best_diff <= tol:
            break
    return best_res


def louvain(adata, n_clusters, key_added='', use_rep="emb", random_seed=2025):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, random_seed=random_seed)
    sc.tl.louvain(adata, resolution=res, random_state=random_seed)
    adata.obs[key_added] = adata.obs['louvain'].astype('int').astype('category')

    n_found = len(adata.obs[key_added].unique())
    if n_found != n_clusters:
        print(f"Warning: Found {n_found} clusters instead of {n_clusters}")
    return adata


def find_optimal_resolution(adata,
                            method: str = 'leiden',
                            resolution_range: Tuple[float, float, float] = (0.1, 2.0, 0.1),
                            use_rep: str = "",
                            random_seed: int = 2025) -> Tuple[float, float]:

    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)

    resolutions = np.arange(*resolution_range)
    modularity_scores = []

    print(f"Searching for optimal resolution for {method}...")

    for res in resolutions:
        temp_key = f'temp_{method}_{res}'
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=res, key_added=temp_key, random_state=random_seed)
        else:  # louvain
            sc.tl.louvain(adata, resolution=res, key_added=temp_key, random_state=random_seed)

        sc.tl.modularity(adata, partition_key=temp_key)
        q = adata.uns['modularity'][temp_key]
        modularity_scores.append(q)

        if temp_key in adata.obs:
            del adata.obs[temp_key]
        if 'modularity' in adata.uns and temp_key in adata.uns['modularity']:
            del adata.uns['modularity'][temp_key]

        print(f"  Resolution: {res:.2f} -> Modularity: {q:.4f}")

    best_idx = np.argmax(modularity_scores)
    best_res = resolutions[best_idx]
    best_q = modularity_scores[best_idx]

    print(f"Optimal resolution: {best_res:.2f} with modularity {best_q:.4f}")
    return best_res, best_q


def leiden_auto(adata,
                resolution: Optional[float] = None,
                key_added: str = 'leiden_clus',
                use_rep: str = "",
                random_seed: int = 2025,
                **kwargs) -> Tuple[sc.AnnData, int, float]:

    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)

    if resolution is None:
        resolution, _ = find_optimal_resolution(
            adata, method='leiden', use_rep=use_rep, random_seed=random_seed, **kwargs
        )

    sc.tl.leiden(adata, resolution=resolution, random_state=random_seed, key_added=key_added)

    n_clusters = adata.obs[key_added].nunique()
    print(f"Leiden clustering with resolution={resolution} found {n_clusters} clusters.")
    return adata



def louvain_auto(adata,
                 resolution: Optional[float] = None,
                 key_added: str = 'louvain_clus',
                 use_rep: str = "",
                 random_seed: int = 2025,
                 **kwargs) -> Tuple[sc.AnnData, int, float]:

    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)

    if resolution is None:
        resolution, _ = find_optimal_resolution(
            adata, method='louvain', use_rep=use_rep, random_seed=random_seed, **kwargs
        )

    sc.tl.louvain(adata, resolution=resolution, random_state=random_seed, key_added=key_added)

    n_clusters = adata.obs[key_added].nunique()
    print(f"Louvain clustering with resolution={resolution} found {n_clusters} clusters.")
    return adata




def mclust_R(adata, n_clusters, use_rep='emb', key_added='mclust', random_seed=2025):

    import os
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    modelNames = 'EEE'

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata



def evaluate_cluster(adata, mode=0,use_rep="emb", GTkey="ground_truth", method_name="mclust"):
    """
    mode:

    """

    if mode == 0:
        ari = metrics.adjusted_rand_score(adata.obs[GTkey], adata.obs[method_name])
        ami = metrics.adjusted_mutual_info_score(adata.obs[GTkey], adata.obs[method_name])
        nmi = metrics.normalized_mutual_info_score(adata.obs[GTkey], adata.obs[method_name])

        #  Purity score
        purity = metrics.accuracy_score(
            adata.obs[GTkey],
            adata.obs[GTkey]
            .groupby(adata.obs[method_name])
            .agg(lambda x: x.value_counts().index[0])
            [adata.obs[method_name]].values
        )

        # Homogeneity, completeness, V-measure
        homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(
            adata.obs[GTkey], adata.obs[method_name]
        )
        print(f"{method_name} ARI:{ari:.4f} NMI:{nmi:.4f} AMI:{ami:.4f} "
              f"purity:{purity:.4f}, homogeneity:{homogeneity:.4f}, completeness:{completeness:.4f}, v_measure:{v_measure:.4f}")
    elif mode == 1:
        sc = metrics.silhouette_score(adata.obsm[use_rep], adata.obs[method_name])
        db = metrics.davies_bouldin_score(adata.obsm[use_rep], adata.obs[method_name])
        print(f"{method_name} SC:{sc} DB:{db}")
    return


def cluster_number_accuracy(n_true: int, m_pred: int) -> float:
    if n_true < 1 or m_pred < 1:
        raise ValueError("n_true and  m_pred must be integers")
    return 1.0 / (1 + abs(m_pred - n_true))


def eval_cluster_number(adata,n,  key_lst=["mclust"]):
    result = []
    for key in key_lst:
        if key in adata.obs:

            m = adata.obs[key].nunique()
            result.append(cluster_number_accuracy(n, m))
    return result


def hm_integration(adata, batch_key='batch_name', embedding_key='emb', harmony_key='emb.Harmony'):

    meta_data = adata.obs[[batch_key]]
    data_mat = adata.obsm[embedding_key]
    vars_use = [batch_key]

    # Run Harmony
    ho = hm.run_harmony(data_mat, meta_data, vars_use)

    # Prepare the Harmony-integrated embedding
    res = pd.DataFrame(ho.Z_corr).T
    res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata.obs.index)

    # Add the Harmony-integrated embedding to adata.obsm
    adata.obsm[harmony_key] = res_df.values
    if isinstance(adata.obsm[harmony_key], pd.DataFrame):
        adata.obsm[harmony_key] = adata.obsm[harmony_key].to_numpy()
    # adata.obsm[harmony_key] = adata.obsm[harmony_key].to_numpy()
    return adata


def eval_batch(adata, output_key, batch_key='batch_name', GT_key="ground_truth", label=False):
    ILISI = hm.compute_lisi(adata.obsm[output_key], adata.obs[[batch_key]], label_colnames=[batch_key])[:, 0]
    median_ILISI = np.median(ILISI)
    avg_ILISI = ILISI.mean()

    print(f"median ILISI:{median_ILISI} ")
    # print(f"mean ILISI:{avg_ILISI}")

    if label:
        CLISI = hm.compute_lisi(adata.obsm[output_key], adata.obs[[ GT_key]], label_colnames=[ GT_key])[:, 0]
        median_CLISI = np.median(CLISI)
        avg_CLISI = CLISI.mean()
        print(f"median CLISI:{median_CLISI} ")
        # print(f"mean CLISI:{avg_CLISI}")
        return ILISI, CLISI
    return ILISI



