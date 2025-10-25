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


def search_res( adata, n_clusters,radius=50, method='leiden', use_rep='norm_emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    ress = []
    best = None
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
    sc.tl.leiden(adata, random_state=0, resolution=end)
    count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    while (count_unique > n_clusters + 2):
        print(count_unique)
        print('太大，继续调整')
        end = end - 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    while (count_unique < n_clusters + 2):
        print(count_unique)
        print('太小，继续调整')
        end = end + 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())

    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))

        if count_unique == n_clusters:
            print('calculate metric ARI')
            # calculate metric ARI
            new_type = refine_label(adata, radius, key='leiden')
            adata.obs['leiden'] = new_type

            ARI = metrics.adjusted_rand_score(adata.obs['leiden'], adata.obs['ground_truth'])
            adata.uns['ARI'] = ARI
            ress.append((res, ARI))
            print('ARI:', ARI)

        if count_unique == n_clusters - 2:
            label = 1
            best = max(ress, key=lambda x: x[1])
            print(best)
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return best[0]


# def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.005, random_seed=2025):
#
#     for res in np.arange(0.1, 5, increment):
#         sc.tl.leiden(adata, random_state=random_seed, resolution=res)
#         if len(adata.obs['leiden'].unique()) > n_clusters:
#             break
#     return res-increment
#
#
# def leiden(adata, n_clusters, key_added='SEDR', use_rep="SEDR", random_seed=2025):
#     sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)
#
#     res = res_search_fixed_clus_leiden(adata, n_clusters, increment=0.005, random_seed=random_seed)
#     # res1 = search_res(adata, n_clusters, use_rep=use_rep)
#     print("leiden res:", res)
#     # print("leiden res:", res1)
#     sc.tl.leiden(adata, random_state=random_seed, resolution=res)
#
#     adata.obs[key_added] = adata.obs['leiden']
#     adata.obs[key_added] = adata.obs[key_added].astype('int')
#     adata.obs[key_added] = adata.obs[key_added].astype('category')
#
#     return adata
#
#
# def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.005, random_seed=2025):
#     for res in np.arange(0.1, 5, increment):
#         sc.tl.louvain(adata, random_state=random_seed, resolution=res)
#         if len(adata.obs['louvain'].unique()) > n_clusters:
#             break
#     return res-increment
#
#
# def louvain(adata, n_clusters, key_added='SEDR', use_rep="SEDR", random_seed=2025):
#     sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)
#     res = res_search_fixed_clus_louvain(adata, n_clusters, increment=0.005, random_seed=random_seed)
#     # res1 = search_res(adata, n_clusters, use_rep=use_rep, method='louvain')
#
#     print("leiden res:", res)
#     # print("leiden res:", res1)
#     sc.tl.louvain(adata, random_state=random_seed, resolution=res)
#
#     adata.obs[key_added] = adata.obs['louvain']
#     adata.obs[key_added] = adata.obs[key_added].astype('int')
#     adata.obs[key_added] = adata.obs[key_added].astype('category')
#
#     return adata



def res_search_fixed_clus_leiden(adata, n_clusters, max_res=3, min_res=0.1, random_seed=2025, tol=0.1, max_iters=50):
    """
    使用二分搜索寻找达到目标聚类数的分辨率
    :param tol: 允许的聚类数量误差
    :param max_iters: 最大迭代次数
    """
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

        # 检查是否达到容差
        if best_diff <= tol:
            break
    return best_res


def leiden(adata, n_clusters, key_added='SEDR', use_rep="SEDR", random_seed=2025):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, random_seed=random_seed)
    sc.tl.leiden(adata, resolution=res, random_state=random_seed)

    # 添加结果并转换类型
    adata.obs[key_added] = adata.obs['leiden'].astype('int').astype('category')

    # 检查聚类结果
    n_found = len(adata.obs[key_added].unique())
    if n_found != n_clusters:
        print(f"Warning: Found {n_found} clusters instead of {n_clusters}")
    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, max_res=3, min_res=0.1, random_seed=2025, tol=0.1, max_iters=50):
    """
    使用二分搜索寻找达到目标聚类数的分辨率
    :param tol: 允许的聚类数量误差
    :param max_iters: 最大迭代次数
    """
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

        # 检查是否达到容差
        if best_diff <= tol:
            break
    return best_res


def louvain(adata, n_clusters, key_added='SEDR', use_rep="SEDR", random_seed=2025):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, random_seed=random_seed)
    sc.tl.louvain(adata, resolution=res, random_state=random_seed)

    # 添加结果并转换类型
    adata.obs[key_added] = adata.obs['louvain'].astype('int').astype('category')

    # 检查聚类结果
    n_found = len(adata.obs[key_added].unique())
    if n_found != n_clusters:
        print(f"Warning: Found {n_found} clusters instead of {n_clusters}")
    return adata


def find_optimal_resolution(adata,
                            method: str = 'leiden',
                            resolution_range: Tuple[float, float, float] = (0.1, 2.0, 0.1),
                            use_rep: str = "SEDR",
                            random_seed: int = 2025) -> Tuple[float, float]:
    """
    通过网格搜索找到使模块度最大化的最佳分辨率。

    参数:
    adata : AnnData对象
    method : str, 'leiden' 或 'louvain'
    resolution_range : tuple, (start, stop, step) 用于生成分辨率搜索范围
    use_rep : str, 用于构建图的特征表示
    random_seed : int, 随机种子

    返回:
    best_res : float, 最佳分辨率
    best_q : float, 对应的模块度值
    """
    # 确保已经计算了邻居图
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)

    resolutions = np.arange(*resolution_range)
    modularity_scores = []

    print(f"Searching for optimal resolution for {method}...")

    for res in resolutions:
        # 使用临时键名进行聚类
        temp_key = f'temp_{method}_{res}'
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=res, key_added=temp_key, random_state=random_seed)
        else:  # louvain
            sc.tl.louvain(adata, resolution=res, key_added=temp_key, random_state=random_seed)

        # 计算模块度
        sc.tl.modularity(adata, partition_key=temp_key)
        q = adata.uns['modularity'][temp_key]
        modularity_scores.append(q)

        # 清理临时结果
        if temp_key in adata.obs:
            del adata.obs[temp_key]
        if 'modularity' in adata.uns and temp_key in adata.uns['modularity']:
            del adata.uns['modularity'][temp_key]

        print(f"  Resolution: {res:.2f} -> Modularity: {q:.4f}")

    # 找到最佳分辨率
    best_idx = np.argmax(modularity_scores)
    best_res = resolutions[best_idx]
    best_q = modularity_scores[best_idx]

    print(f"Optimal resolution: {best_res:.2f} with modularity {best_q:.4f}")
    return best_res, best_q


def leiden_auto(adata,
                resolution: Optional[float] = None,
                key_added: str = 'leiden_clus',
                use_rep: str = "SEDR",
                random_seed: int = 2025,
                **kwargs) -> Tuple[sc.AnnData, int, float]:
    """
    使用Leiden算法进行全自动无监督聚类。
    如果未提供resolution，则自动寻找使模块度最大化的最佳分辨率。

    参数:
    adata : AnnData对象
    resolution : float, 可选。如果为None，则自动寻找最佳分辨率
    key_added : str, 结果存储的键名
    use_rep : str, 用于构建图的特征表示
    random_seed : int, 随机种子
    **kwargs : 传递给find_optimal_resolution的额外参数

    返回:
    adata : 添加了聚类结果的AnnData对象
    n_clusters : 发现的簇数量
    resolution_used : 实际使用的分辨率
    """
    # 确保已经计算了邻居图
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)

    # 自动寻找最佳分辨率（如果未提供）
    if resolution is None:
        resolution, _ = find_optimal_resolution(
            adata, method='leiden', use_rep=use_rep, random_seed=random_seed, **kwargs
        )

    # 使用选定分辨率运行Leiden算法
    sc.tl.leiden(adata, resolution=resolution, random_state=random_seed, key_added=key_added)

    # 获取聚类结果
    n_clusters = adata.obs[key_added].nunique()
    print(f"Leiden clustering with resolution={resolution} found {n_clusters} clusters.")
    return adata
    # return adata, n_clusters, resolution


def louvain_auto(adata,
                 resolution: Optional[float] = None,
                 key_added: str = 'louvain_clus',
                 use_rep: str = "SEDR",
                 random_seed: int = 2025,
                 **kwargs) -> Tuple[sc.AnnData, int, float]:
    """
    使用Louvain算法进行全自动无监督聚类。
    如果未提供resolution，则自动寻找使模块度最大化的最佳分辨率。
    """
    # 确保已经计算了邻居图
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)

    # 自动寻找最佳分辨率（如果未提供）
    if resolution is None:
        resolution, _ = find_optimal_resolution(
            adata, method='louvain', use_rep=use_rep, random_seed=random_seed, **kwargs
        )

    # 使用选定分辨率运行Louvain算法
    sc.tl.louvain(adata, resolution=resolution, random_state=random_seed, key_added=key_added)

    # 获取聚类结果
    n_clusters = adata.obs[key_added].nunique()
    print(f"Louvain clustering with resolution={resolution} found {n_clusters} clusters.")
    return adata
    # return adata, n_clusters, resolution



def mclust_R(adata, n_clusters, use_rep='emb', key_added='mclust', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import os
    # os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'
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
            # 将聚类结果按纯度最大原则映射到真实标签
            adata.obs[GTkey]
            .groupby(adata.obs[method_name])
            .agg(lambda x: x.value_counts().index[0])
            [adata.obs[method_name]].values
        )

        # Homogeneity, completeness, V-measure（一行可得）
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
# def eval_batch(adata, output_key, batch_key='batch_name', GT_key="ground_truth", label=False):
#
#
#     ILISI = hm.compute_lisi(adata.obsm[output_key], adata.obs[[batch_key]], label_colnames=[batch_key])[:, 0]
#     median_ILISI = np.median(ILISI)
#     avg_ILISI = ILISI.mean()
#
#     print(f"median ILISI:{median_ILISI} ")
#     print(f"mean ILISI:{avg_ILISI}")
#
#     if label:
#         CLISI = hm.compute_lisi(adata.obsm[output_key], adata.obs[[ GT_key]], label_colnames=[ GT_key])[:, 0]
#         median_CLISI = np.median(CLISI)
#         avg_CLISI = CLISI.mean()
#         print(f"median CLISI:{median_CLISI} ")
#         print(f"mean CLISI:{avg_CLISI}")
#         return ILISI, CLISI
#     return ILISI



global_results = {}

def CLuster_auto(adata, n_clusters, model_name="", dataset_name="", use_rep="SEDR", random_seed=2025, label=True):
    if dataset_name  not in global_results:
        global_results[dataset_name] = {}

    # 聚类分析并存储结果
    clustering_methods = [
        # ("dtne", dtne),
        # ('KMeans', KMeans(n_clusters=n_clusters, random_state=random_seed)),
        # ('KMedoids', KMedoids(n_clusters=n_clusters, random_state=random_seed)),
        # ('mclust', mclust_R),
        ('louvain', louvain_auto),
        ('leiden', leiden_auto),

    ]

    for method_name, method in clustering_methods:
        # print(f"cluster {method_name}...")

        if method_name == 'mclust':
            method(adata, n_clusters, use_rep=use_rep, key_added=method_name, random_seed=random_seed)
        elif method_name in ['louvain', 'leiden']:
            method(adata, n_clusters, use_rep=use_rep, key_added=method_name, random_seed=random_seed)
        else:
            labels = method.fit_predict(adata.obsm[use_rep])
            adata.obs[method_name] = labels
            adata.obs[method_name] = adata.obs[method_name].astype('int')
            adata.obs[method_name] = adata.obs[method_name].astype('category')

        if label:
            # 计算并存储ARI和NMI
            ari = metrics.adjusted_rand_score(adata.obs["ground_truth"], adata.obs[method_name])
            ami = metrics.adjusted_mutual_info_score(adata.obs["ground_truth"], adata.obs[method_name])
            nmi = metrics.normalized_mutual_info_score(adata.obs["ground_truth"], adata.obs[method_name], average_method="geometric")

            #  Purity score
            purity = metrics.accuracy_score(
                adata.obs["ground_truth"],
                # 将聚类结果按纯度最大原则映射到真实标签
                adata.obs["ground_truth"]
                .groupby(adata.obs[method_name])
                .agg(lambda x: x.value_counts().index[0])
                [adata.obs[method_name]].values
            )

            # Homogeneity, completeness, V-measure（一行可得）
            homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(
                adata.obs["ground_truth"], adata.obs[method_name]
            )

            print(f"{method_name} ARI:{ari:.4f} NMI:{nmi:.4f} AMI:{ami:.4f} "
                  f"purity:{purity:.4f}, homogeneity:{homogeneity:.4f}, completeness:{completeness:.4f}, v_measure:{v_measure:.4f}")
            # 存储到全局字典
            global_results[dataset_name][method_name] = {'ARI': ari, 'NMI': nmi, 'AMI': ami,
                                        "purity":purity, "homogeneity":homogeneity,"completeness":completeness, 'v_measure':v_measure, }
        else:
            sc = metrics.silhouette_score(adata.obsm[use_rep], adata.obs[method_name])
            db = metrics.davies_bouldin_score(adata.obsm[use_rep], adata.obs[method_name])
            print(f"{method_name} SC:{sc} DB:{db}")
            global_results[dataset_name][method_name] =  {'SC': sc, 'DB':db }


def CLuster(adata, n_clusters, model_name="", dataset_name="", use_rep="SEDR", random_seed=2025, label=True):
    if dataset_name  not in global_results:
        global_results[dataset_name] = {}

    # 聚类分析并存储结果
    clustering_methods = [
        # ("dtne", dtne),
        ('KMeans', KMeans(n_clusters=n_clusters, random_state=random_seed)),
        ('KMedoids', KMedoids(n_clusters=n_clusters, random_state=random_seed)),
        # ('mclust', mclust_R),
        ('louvain', louvain),
        ('leiden', leiden),

    ]

    for method_name, method in clustering_methods:
        # print(f"cluster {method_name}...")

        if method_name == 'mclust':
            method(adata, n_clusters, use_rep=use_rep, key_added=method_name, random_seed=random_seed)
        elif method_name in ['louvain', 'leiden']:
            method(adata, n_clusters, use_rep=use_rep, key_added=method_name, random_seed=random_seed)
        else:
            labels = method.fit_predict(adata.obsm[use_rep])
            adata.obs[method_name] = labels
            adata.obs[method_name] = adata.obs[method_name].astype('int')
            adata.obs[method_name] = adata.obs[method_name].astype('category')

        if label:
            # 计算并存储ARI和NMI
            ari = metrics.adjusted_rand_score(adata.obs["ground_truth"], adata.obs[method_name])
            ami = metrics.adjusted_mutual_info_score(adata.obs["ground_truth"], adata.obs[method_name])
            nmi = metrics.normalized_mutual_info_score(adata.obs["ground_truth"], adata.obs[method_name], average_method="geometric")

            #  Purity score
            purity = metrics.accuracy_score(
                adata.obs["ground_truth"],
                # 将聚类结果按纯度最大原则映射到真实标签
                adata.obs["ground_truth"]
                .groupby(adata.obs[method_name])
                .agg(lambda x: x.value_counts().index[0])
                [adata.obs[method_name]].values
            )

            # Homogeneity, completeness, V-measure（一行可得）
            homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(
                adata.obs["ground_truth"], adata.obs[method_name]
            )

            print(f"{method_name} ARI:{ari:.4f} NMI:{nmi:.4f} AMI:{ami:.4f} "
                  f"purity:{purity:.4f}, homogeneity:{homogeneity:.4f}, completeness:{completeness:.4f}, v_measure:{v_measure:.4f}")
            # 存储到全局字典
            global_results[dataset_name][method_name] = {'ARI': ari, 'NMI': nmi, 'AMI': ami,
                                        "purity":purity, "homogeneity":homogeneity,"completeness":completeness, 'v_measure':v_measure, }
        else:
            sc = metrics.silhouette_score(adata.obsm[use_rep], adata.obs[method_name])
            db = metrics.davies_bouldin_score(adata.obsm[use_rep], adata.obs[method_name])
            print(f"{method_name} SC:{sc} DB:{db}")
            global_results[dataset_name][method_name] =  {'SC': sc, 'DB':db }


# global_results = {}
def CLuster_(adata, n_clusters, model_name="", dataset_name="", use_rep="SEDR", random_seed=2025, label=True):
    if dataset_name  not in global_results:
        global_results[dataset_name] = {}

    # 聚类分析并存储结果
    clustering_methods = [
        # ("dtne", dtne),
        ('KMeans', KMeans(n_clusters=n_clusters, random_state=random_seed)),
        ('KMedoids', KMedoids(n_clusters=n_clusters, random_state=random_seed)),
        ('mclust', mclust_R),
        ('louvain', louvain),
        ('leiden', leiden),

    ]

    for method_name, method in clustering_methods:
        # print(f"cluster {method_name}...")

        if method_name == 'mclust':
            method(adata, n_clusters, use_rep=use_rep, key_added=method_name, random_seed=random_seed)
        elif method_name in ['louvain', 'leiden']:
            method(adata, n_clusters, use_rep=use_rep, key_added=method_name, random_seed=random_seed)
        else:
            labels = method.fit_predict(adata.obsm[use_rep])
            adata.obs[method_name] = labels
            adata.obs[method_name] = adata.obs[method_name].astype('int')
            adata.obs[method_name] = adata.obs[method_name].astype('category')

        if label:
            # 计算并存储ARI和NMI
            ari = metrics.adjusted_rand_score(adata.obs["ground_truth"], adata.obs[method_name])
            ami = metrics.adjusted_mutual_info_score(adata.obs["ground_truth"], adata.obs[method_name])
            nmi = metrics.normalized_mutual_info_score(adata.obs["ground_truth"], adata.obs[method_name], average_method="geometric")

            #  Purity score
            purity = metrics.accuracy_score(
                adata.obs["ground_truth"],
                # 将聚类结果按纯度最大原则映射到真实标签
                adata.obs["ground_truth"]
                .groupby(adata.obs[method_name])
                .agg(lambda x: x.value_counts().index[0])
                [adata.obs[method_name]].values
            )

            # Homogeneity, completeness, V-measure（一行可得）
            homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(
                adata.obs["ground_truth"], adata.obs[method_name]
            )

            print(f"{method_name} ARI:{ari:.4f} NMI:{nmi:.4f} AMI:{ami:.4f} "
                  f"purity:{purity:.4f}, homogeneity:{homogeneity:.4f}, completeness:{completeness:.4f}, v_measure:{v_measure:.4f}")
            # 存储到全局字典
            global_results[dataset_name][method_name] = {'ARI': ari, 'NMI': nmi, 'AMI': ami,
                                        "purity":purity, "homogeneity":homogeneity,"completeness":completeness, 'v_measure':v_measure, }
        else:
            sc = metrics.silhouette_score(adata.obsm[use_rep], adata.obs[method_name])
            db = metrics.davies_bouldin_score(adata.obsm[use_rep], adata.obs[method_name])
            print(f"{method_name} SC:{sc} DB:{db}")
            global_results[dataset_name][method_name] =  {'SC': sc, 'DB':db }


def save_results_to_csv(filename="clustering_results.csv", save=True):
    import pandas as pd

    # 收集所有可能出现的指标名
    all_metrics = set()
    for dataset, models in global_results.items():
        for m in models.values():
            all_metrics.update(m.keys())

    # 逐数据集构造行
    rows = []
    for dataset, models in global_results.items():
        row = {"dataset": dataset}
        for method, metrics_dict in models.items():
            for metric, value in metrics_dict.items():
                row[f"{method}_{metric}"] = value
        rows.append(row)

    # 构造 DataFrame
    df = pd.DataFrame(rows).set_index("dataset")

    # # ---------- 新增：计算并打印均值 ----------
    # mean_series = df.mean(axis=0)          # 每一列的均值
    # print("\n=== Average across datasets ===")
    # for col in mean_series.index:
    #     print(f"{col}: {mean_series[col]:.4f}")
    # print("================================\n")
    # # ----------------------------------------
    #
    # # 把均值行附加到末尾
    # mean_row = {"dataset": "mean"}
    # mean_row.update(mean_series.to_dict())
    # df = pd.concat([df, pd.DataFrame([mean_row]).set_index("dataset")])

    # 保存
    if save:
        df.to_csv(filename, float_format="%.4f")
        print(f"All results saved to {filename} (with 'mean' row)")
    else:
        print(f"Done!")


# def save_results_to_csv(filename="clustering_results.csv"):
#     # 创建空DataFrame
#     df = pd.DataFrame()
#
#     # 遍历每个数据集
#     for dataset, models in global_results.items():
#         row_data = {}
#         # 遍历该数据集下的所有模型
#         for model_name, metrics_dict in models.items():
#             # 添加两列: [model_name]_ARI 和 [model_name]_NMI
#             row_data[f"{model_name}_ARI"] = metrics_dict['ARI']
#             row_data[f"{model_name}_NMI"] = metrics_dict['NMI']
#             row_data[f"{model_name}_AMI"] = metrics_dict['AMI']
#
#         # 将当前数据集的结果添加到DataFrame
#         df = pd.concat([df, pd.DataFrame(row_data, index=[dataset])])
#
#     # 保存到CSV（空值自动填充为NaN）
#     df.to_csv(filename, float_format="%.4f")
#     print(f"Results saved to {filename}")




# # 定义测试函数
# def test_functions():
#     # 创建一个随机的 AnnData 对象作为测试输入
#     np.random.seed(2025)  # 确保生成的随机数据一致
#     X = np.random.rand(100, 10)  # 生成 100 个样本，每个样本有 10 个特征
#     adata = anndata.AnnData(X=X)
#
#     # 定义测试参数
#     n_clusters = 5
#     random_seed = 2025
#
#     # 首先计算 neighbors 图结构
#     sc.pp.neighbors(adata, random_state=random_seed)
#
#     # 第一次运行函数
#     print("第一次运行函数")
#     res1 = res_search_fixed_clus_louvain(adata.copy(), n_clusters, random_seed=random_seed)
#     adata1 = louvain(adata.copy(), n_clusters, random_seed=random_seed)
#
#     # 第二次运行函数
#     print("第二次运行函数")
#     res2 = res_search_fixed_clus_louvain(adata.copy(), n_clusters, random_seed=random_seed)
#     adata2 = louvain(adata.copy(), n_clusters, random_seed=random_seed)
#
#     # 检查结果是否一致
#     print("检查 res_search_fixed_clus_louvain 函数的输出是否一致：")
#     print("第一次运行结果:", res1)
#     print("第二次运行结果:", res2)
#     assert res1 == res2, "res_search_fixed_clus_louvain 函数的输出不一致！"
#
#     print("检查 louvain 函数的输出是否一致：")
#     print("第一次运行结果的 louvain 分群结果:", adata1.obs['SEDR'].values)
#     print("第二次运行结果的 louvain 分群结果:", adata2.obs['SEDR'].values)
#     assert np.array_equal(adata1.obs['SEDR'].values, adata2.obs['SEDR'].values), "louvain 函数的输出不一致！"
#
#     print("测试通过，两个函数在输入一定、随机种子一定的情况下输出一致！")
#
# # 运行测试函数
# test_functions()