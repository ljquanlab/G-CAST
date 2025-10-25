import numpy as np
import torch
import scipy.sparse as sp
from scipy.special import softmax
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag



##### generate n
def generate_adj_mat(adata, include_self=False, n=6, key='spatial'):
    from sklearn import metrics
    assert key in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm[key])
    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0
    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

##### normalze graph
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adj_prob(adata,adj, name="feat"):
    np.fill_diagonal(adj, 0)

    expression = adata.obsm[name]  # shape: [N, d]

    # 计算欧氏距离矩阵（N x N）
    euclidean_distances = cdist(expression, expression, metric='euclidean')

    # 初始化概率邻接矩阵
    N = adj.shape[0]
    edge_probabilities = np.zeros((N, N), dtype=np.float32)

    # 对每一行的邻居做 softmax
    for i in range(N):
        neighbors = adj[i] == 1
        if np.any(neighbors):
            dists = euclidean_distances[i][neighbors]
            # weights = -dists
            weights = -np.log((dists) + 1e-6)
            probs = softmax(weights)  # 归一化成概率
            # weights = 1 / (dists + 1e-6)  # 距离越近，权重越大
            # probs = weights / weights.sum()
            edge_probabilities[i][neighbors] = probs
        # 可选：保留自环
        edge_probabilities[i, i] = 1  # 或设为最大prob的一小部分
    # 保存结果
    edge_probabilities = (edge_probabilities + edge_probabilities.T) / 2
    adata.obsm[f'edge_prob_{name}'] = edge_probabilities

    return edge_probabilities




def graph_construction(adata, n=6, dmax=50, mode='KNN', emb=True):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax)

    edge_prob_feat = adj_prob(adata, adj_m1, name="X_pca")
    if emb:
        edge_prob_img = adj_prob(adata, adj_m1, name="img")

    adj_m1 = sp.coo_matrix(adj_m1) # 将邻接矩阵转化为稀疏矩阵的形式
    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)

    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    if emb:
        graph_dict = {
            "adj_norm": adj_norm_m1,
            "adj_label": adj_label_m1.coalesce(),
            "adj": adj_label_m1.coalesce().to_dense().bool().float(),
            "norm_value": norm_m1,
            "edge_prob_feat": edge_prob_feat,
            "edge_prob_img": edge_prob_img,    #
        }
    else:
        graph_dict = {
            "adj_norm": adj_norm_m1,
            "adj_label": adj_label_m1.coalesce(),
            "adj": adj_label_m1.coalesce().to_dense().bool().float(),
            "norm_value": norm_m1,
            "edge_prob_feat": edge_prob_feat,
        }

    return graph_dict




# def graph_construction_1(adata, n=6, dmax=50, mode='KNN'):
#     if mode == 'KNN':
#         adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
#     else:
#         adj_m1 = generate_adj_mat_1(adata, dmax)
#
#     adj_m1 = sp.coo_matrix(adj_m1) # 将邻接矩阵转化为稀疏矩阵的形式
#     # Store original adjacency matrix (without diagonal entries) for later
#     adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
#     adj_m1.eliminate_zeros()
#
#     # Some preprocessing
#     adj_norm_m1 = preprocess_graph(adj_m1)
#
#     adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
#     adj_m1 = adj_m1.tocoo()
#     shape = adj_m1.shape
#     values = adj_m1.data
#
#     indices = np.stack([adj_m1.row, adj_m1.col])
#     adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)
#
#     norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
#
#     graph_dict = {
#         "adj_norm": adj_norm_m1,
#         "adj_label": adj_label_m1.coalesce(),
#         "adj": adj_label_m1.coalesce().to_dense().bool().float(),
#         "norm_value": norm_m1,
#     }
#
#     return graph_dict


def block_diag_sparse(*arrs):
        bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
        if bad_args:
            raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

        list_shapes = [a.shape for a in arrs]
        list_indices = [a.coalesce().indices().clone() for a in arrs]
        list_values = [a.coalesce().values().clone() for a in arrs]

        r_start = 0
        c_start = 0
        for i in range(len(arrs)):
            list_indices[i][0, :] += r_start
            list_indices[i][1, :] += c_start

            r_start += list_shapes[i][0]
            c_start += list_shapes[i][1]

        indices = torch.concat(list_indices, axis=1)
        values = torch.concat(list_values)
        shapes = torch.tensor(list_shapes).sum(axis=0)

        out = torch.sparse_coo_tensor(indices, values, (shapes[0], shapes[1]))
        return out


def create_block_diag(*matrices):
    # 确定总行数和列数
    total_rows = sum(matrix.shape[0] for matrix in matrices)
    total_cols = sum(matrix.shape[1] for matrix in matrices)

    # 创建全零矩阵
    block_diag = np.zeros((total_rows, total_cols))

    # 填充块对角矩阵
    row_offset = 0
    col_offset = 0
    for matrix in matrices:
        rows, cols = matrix.shape
        block_diag[row_offset:row_offset + rows, col_offset:col_offset + cols] = matrix
        row_offset += rows
        col_offset += cols

    return block_diag


def combine_graph_dict(*dicts):
    # 检查是否至少有一个字典
    if not dicts:
        raise ValueError("At least one dictionary must be provided")

    # 处理稀疏矩阵的块对角拼接
    tmp_adj_norm = block_diag_sparse(*(d['adj_norm'] for d in dicts))  # 解包多个字典的adj_norm
    tmp_adj_label = block_diag_sparse(*(d['adj_label'] for d in dicts))  # 解包多个字典的adj_label
    # tmp_adj_prob_feat_sparse = block_diag_sparse(*(d["adj_prob_feat_sparse"] for d in dicts))
    # tmp_adj_prob_img_sparse = block_diag_sparse(*(d["adj_prob_img_sparse"] for d in dicts))

    # 处理稠密矩阵的块对角拼接
    tmp_edge_feat_prob = block_diag(*(d['edge_prob_feat'] for d in dicts))

    edge_prob_img_list = []
    # edge_prob_img_mask_list = []
    for d in dicts:
        feat = d['edge_prob_feat']  # shape: [num_edges, dim]
        if 'edge_prob_img' in d and d['edge_prob_img'] is not None:
            edge_prob_img_list.append(d['edge_prob_img'])
            # edge_prob_img_mask_list.append(np.ones_like(d['edge_prob_img'], dtype=bool))
        else:
            edge_prob_img_list.append(np.zeros_like(feat))
            # edge_prob_img_mask_list.append(np.zeros_like(feat, dtype=bool))
    tmp_edge_img_prob = block_diag(*edge_prob_img_list)
    # edge_prob_img_mask = block_diag(*edge_prob_img_mask_list)

    # 假设所有字典的adj矩阵形状一致，用于生成零矩阵
    tmp_adj = block_diag(*(d["adj"] for d in dicts))

    # 计算多个norm_value的平均值
    norm_values = [d['norm_value'] for d in dicts]
    avg_norm_value = np.mean(norm_values)

    # 构建结果字典
    graph_dict = {
        "adj_norm": tmp_adj_norm.coalesce(),
        "adj_label": tmp_adj_label.coalesce(),
        "norm_value": avg_norm_value,
        "adj": tmp_adj,
        # "adj_prob_feat_sparse": tmp_adj_prob_feat_sparse.coalesce(),
        # "adj_prob_img_sparse": tmp_adj_prob_img_sparse.coalesce(),
        "edge_prob_img": tmp_edge_img_prob,
        # "img_mask": edge_prob_img_mask,
        "edge_prob_feat": tmp_edge_feat_prob,
    }
    return graph_dict
