#
import os
import torch
import numpy as np
import scanpy as sc
import random

def adata_preprocess(adata_vis, min_cells=50, min_counts=10, pca_n_comps=200):
    adata_vis.layers['count'] = adata_vis.X.toarray()
    sc.pp.filter_genes(adata_vis, min_cells=min_cells)
    sc.pp.filter_genes(adata_vis, min_counts=min_counts)

#     adata_vis.obs['mean_exp'] = adata_vis.X.toarray().mean(axis=1)
#     adata_vis.var['mean_exp'] = adata_vis.X.toarray().mean(axis=0)
#
#     # Load scRNA-seq data
#     adata_ref = sc.read_h5ad('/home/xuhang/disco_500t/Projects/spTrans/data/reference_data/GSE144136_DLPFC/raw/processed_raw.h5ad')
#     adata_ref.obs['mean_exp'] = adata_ref.X.toarray().mean(axis=1)
#     adata_ref.var['mean_exp'] = adata_ref.X.toarray().mean(axis=0)
#     common_genes = np.intersect1d(adata_vis.var.index, adata_ref.var.index)
#     adata_vis = adata_vis[:, common_genes]
#     adata_ref = adata_ref[:, common_genes]
#     adata_vis.var['ref_mean_exp'] = adata_ref.var['mean_exp']
#     adata_vis.var['ratio'] = np.log10(adata_vis.var['mean_exp'] / adata_vis.var['ref_mean_exp']+1)
#     adata_vis.var['selected'] = adata_vis.var['ratio'] < 1.5
#     remain_genes = adata_vis.var[adata_vis.var['selected']==True].index.tolist()
#     adata_vis = adata_vis[:, remain_genes]
#
#

    sc.pp.normalize_total(adata_vis, target_sum=1e6)
    # sc.pp.log1p(adata_vis)
    sc.pp.highly_variable_genes(adata_vis, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata_vis = adata_vis[:, adata_vis.var['highly_variable'] == True]
    sc.pp.scale(adata_vis)

    from sklearn.decomposition import PCA
    adata_X = PCA(n_components=pca_n_comps, random_state=42).fit_transform(adata_vis.X)
    adata_vis.obsm['X_pca'] = adata_X
    return adata_vis


# 设置随机种子函数
def set_seed(random_seed):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_default_dtype(torch.float64)


class EarlyStopping:
    def __init__(self,
                 patience: int = 5,
                 delta: float = 0.0,
                 verbose: bool = False,
                 path: str = 'checkpoint.pth',
                 mode: str = 'save'):  # 'save' 保存到磁盘；'return' 保存在内存
        """
        Args:
            patience (int): 容忍的 epoch 数量
            delta (float): 最小改进值
            verbose (bool): 是否打印日志
            path (str): 模型/断点保存路径
            mode (str): 'save' 保存到磁盘（默认）
                        'return' 仅在内存中缓存 best_state
        """
        assert mode in ('save', 'return'), "mode 必须是 'save' 或 'return'"
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.mode = mode

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None  # 存储完整 checkpoint（mode='return' 时使用）

    def __call__(self, val_loss, model, optimizer=None, epoch=None):
        """
        Args:
            val_loss (float): 当前验证集损失
            model (torch.nn.Module): 当前模型
            optimizer (torch.optim.Optimizer, optional): 优化器，可选
            epoch (int, optional): 当前 epoch，可选
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            # 有改进
            if self.verbose:
                if self.best_loss is None:
                    print(f"Initial validation loss set to {val_loss:.6f}. Saving checkpoint...")
                else:
                    print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving checkpoint...")

            # 构建 checkpoint 包含完整训练状态
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'epoch': epoch if epoch is not None else -1,
                'best_loss': val_loss
            }
            # print( 'rng_state:', torch.get_rng_state())
            # print(len(torch.get_rng_state()))
            #
            # print("CUDA count:", torch.cuda.device_count())
            # print("CUDA current device:", torch.cuda.current_device())
            # print("CUDA RNG states:", torch.cuda.get_rng_state_all())

            if self.mode == 'save':
                torch.save(checkpoint, self.path)
                if self.verbose:
                    print(f"Checkpoint saved to {self.path}")
            elif self.mode == 'return':
                # 存到内存中（可手动保存或加载）
                self.best_state = {k: v for k, v in checkpoint.items()}
                if self.verbose:
                    print("Checkpoint cached in memory.")

            self.best_loss = val_loss
            self.counter = 0
        else:
            # 没有改进
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
