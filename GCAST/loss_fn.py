import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F


def stable(probs, seed=42, epsilon=1e-8):
    quantized = (probs * 1e8).round()

    # 找到最大值和候选位置
    max_vals, _ = torch.max(quantized, dim=1, keepdim=True)
    candidates = quantized == max_vals

    # 为每个候选位置生成固定随机分数
    torch.manual_seed(seed)
    scores = torch.rand_like(probs)

    # 只保留候选位置的分数
    masked_scores = torch.where(candidates, scores, torch.tensor(-torch.inf).to(probs.device))

    # 选择最高分数（平局时随机但确定）
    # return masked_scores.argmax(dim=1).cpu().numpy()
    return masked_scores

def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn


class InfoNCEWithAdj(nn.Module):
    def __init__(self, temperature=0.1, normalize=True):
        """
        支持邻接矩阵的InfoNCE损失函数

        Args:
            temperature (float, optional): 温度系数，缩放相似度。默认0.1。
            normalize (bool, optional): 是否对输入进行L2归一化。默认True。
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, query, key, adj):
        """
        前向传播计算损失
        Args:
            query (torch.Tensor): 查询向量，形状为(batch_size, embedding_dim)
            key (torch.Tensor): 关键向量，形状与query相同
            adj (torch.Tensor): 邻接矩阵，形状为(batch_size, batch_size)，记录样本间是否为正样本对
        Returns:
            torch.Tensor: 计算得到的对比损失
        """
        if self.normalize:
            query = F.normalize(query, p=2, dim=1)
            key = F.normalize(key, p=2, dim=1)

        # 计算相似度矩阵 (batch_size, batch_size)
        logits = torch.matmul(query, key.T) / self.temperature

        # 生成正样本掩码 (排除自身需设置adj[i,i]=0)
        mask = adj.to(query.dtype)  # 确保类型匹配

        # 计算对数softmax (数值稳定)
        log_softmax = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # 计算正样本的加权损失
        loss = - (mask * log_softmax).sum(dim=1).mean()
        return loss



class InfoNCEWithAdj(nn.Module):
    def __init__(self, temperature=0.1, normalize=True, chunk_size=512):
        """
        完全避免原地操作的稳定InfoNCE损失函数
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.chunk_size = chunk_size

    def _normalize_sparse(self, x):
        """稀疏矩阵L2归一化"""
        if not x.is_sparse:
            return F.normalize(x, p=2, dim=1)

        # 计算稀疏矩阵的行范数
        x_squared = torch.sparse_coo_tensor(
            x._indices(),
            x._values() ** 2,
            x.size()
        )
        row_sums = torch.sparse.sum(x_squared, dim=1).to_dense()
        row_norms = row_sums.sqrt().clamp(min=1e-8)

        # 归一化值
        indices = x._indices()
        values = x._values()
        row_idx = indices[0]
        scale = 1.0 / row_norms[row_idx]
        values_norm = values * scale

        return torch.sparse_coo_tensor(
            indices, values_norm, x.size(),
            device=x.device, dtype=x.dtype
        )

    def forward(self, query, key, adj):
        """
        避免任何原地操作的分块计算
        """
        # 1. 归一化处理
        if self.normalize:
            if hasattr(query, 'is_sparse') and query.is_sparse:
                query = self._normalize_sparse(query)
            else:
                query = F.normalize(query, p=2, dim=1)

            if hasattr(key, 'is_sparse') and key.is_sparse:
                key = self._normalize_sparse(key)
            else:
                key = F.normalize(key, p=2, dim=1)

        # 2. 转换为稠密矩阵
        if hasattr(query, 'is_sparse') and query.is_sparse:
            query = query.to_dense()
        if hasattr(key, 'is_sparse') and key.is_sparse:
            key = key.to_dense()
        adj = adj.to(query.dtype)

        N = query.size(0)
        chunk_size = min(self.chunk_size, N)  # 确保不超过总样本数

        # 3. 使用列表存储中间结果（避免原地操作）
        global_max_list = []
        global_sum_exp_list = []

        # 4. 第一轮遍历：计算全局最大值
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            q_chunk = query[i:end]

            # 计算当前块相似度
            logits_chunk = torch.matmul(q_chunk, key.T) / self.temperature

            # 计算当前块的最大值
            current_max = logits_chunk.max(dim=1, keepdim=True).values
            global_max_list.append(current_max)

        # 合并最大值
        global_max = torch.cat(global_max_list, dim=0)

        # 5. 第二轮遍历：计算指数和
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            q_chunk = query[i:end]

            # 计算当前块相似度
            logits_chunk = torch.matmul(q_chunk, key.T) / self.temperature

            # 使用全局最大值稳定计算
            exp_chunk = torch.exp(logits_chunk - global_max[i:end])
            current_sum_exp = exp_chunk.sum(dim=1, keepdim=True)
            global_sum_exp_list.append(current_sum_exp)

        # 合并指数和
        global_sum_exp = torch.cat(global_sum_exp_list, dim=0)

        # 6. 计算全局分母
        global_denom = global_max + torch.log(global_sum_exp)

        # 7. 第三轮遍历：计算损失
        total_loss = 0.0
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            q_chunk = query[i:end]
            adj_chunk = adj[i:end]

            # 计算当前块相似度
            logits_chunk = torch.matmul(q_chunk, key.T) / self.temperature

            # 计算log_softmax
            log_softmax_chunk = logits_chunk - global_denom[i:end]

            # 计算当前块损失
            pos_logits = (adj_chunk * log_softmax_chunk).sum(dim=1)
            chunk_loss = -pos_logits.mean()
            total_loss += chunk_loss * (end - i)

        return total_loss / N



class Contrast(nn.Module):
    def __init__(self, tempture=0.5):
        super(Contrast, self).__init__()
        self.tempture = tempture


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.exp(torch.mm(z1, z2.t()) / self.tempture)

    def forward(self, z_mp, z_sc, pos):
        # z_proj_mp = self.proj(z_mp)
        # z_proj_sc = self.proj(z_sc)
        z_proj_mp = z_mp
        z_proj_sc = z_sc
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2mp = self.sim(z_proj_mp, z_proj_mp)
        matrix_sc2sc = self.sim(z_proj_sc, z_proj_sc)

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()

        matrix_mp2mp = matrix_mp2mp / (torch.sum(matrix_mp2mp, dim=1).view(-1, 1) + 1e-8)
        lori_mp2mp = -torch.log(matrix_mp2mp.mul(pos).sum(dim=-1)).mean()

        matrix_sc2sc = matrix_sc2sc / (torch.sum(matrix_sc2sc, dim=1).view(-1, 1) + 1e-8)
        lori_sc2sc = -torch.log(matrix_sc2sc.mul(pos).sum(dim=-1)).mean()

        total_loss = lori_mp + lori_sc + lori_mp2mp + lori_sc2sc
        return total_loss



def gcn_loss(preds,preds_img, preds_feat, labels_img, labels_feat,  labels, mu, logvar, n_nodes, norm):
    # print(labels.shape)
    # print(labels_img.shape)
    # print(labels_feat.shape)
    #
    # print(preds.shape)
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    # cost_feat = norm * F.binary_cross_entropy_with_logits(preds, labels_feat)
    cost_img = norm*F.binary_cross_entropy_with_logits(preds_img, labels_img)
    cost_feat = norm*F.binary_cross_entropy_with_logits(preds_feat, labels_feat)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    cost_all = 2*cost + cost_img + cost_feat
    # return cost + KLD
    return cost_all+KLD

def kl_loss(mu, logvar, n_nodes):
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return KLD




class WeightedMultiLoss(nn.Module):
    def __init__(self, weights=3):
        super().__init__()
        # 三个可学习的log_vars
        self.log_vars = nn.Parameter(torch.zeros(weights))

    def forward(self, loss_contrastive, loss_kl, loss_reconstruct):
        losses = [loss_contrastive, loss_kl, loss_reconstruct]
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]  # 相当于 log σ_i
        return total_loss


class GatedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([10, 0.01, 0.2]))  # 初始 λ = [1/3, 1/3, 1/3]

    def forward(self, loss_contrastive, loss_kl, loss_reconstruct):
        # weights = F.softmax(self.logits, dim=0)
        weights = self.logits
        total_loss = weights[0]*loss_contrastive + weights[1]*loss_kl + weights[2]*loss_reconstruct
        return total_loss


# class GatedLoss(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, 8),
#             nn.ReLU(),
#             nn.Linear(8, 3),     # 三个loss
#             nn.Softmax(dim=-1)    # 输出为[λ1, λ2, λ3]
#         )
#         # self.gate[-2].bias.data = torch.tensor([10, 0.01, 0.2])
#
#     def forward(self, x_feat, loss_contrastive, loss_kl, loss_reconstruct):
#         weights = self.gate(x_feat)
#         total_loss = weights[0]*loss_contrastive + weights[1]*loss_kl + weights[2]*loss_reconstruct
#         return total_loss