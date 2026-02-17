import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F


def stable(probs, seed=42, epsilon=1e-8):
    quantized = (probs * 1e8).round()
    max_vals, _ = torch.max(quantized, dim=1, keepdim=True)
    candidates = quantized == max_vals
    torch.manual_seed(seed)
    scores = torch.rand_like(probs)
    masked_scores = torch.where(candidates, scores, torch.tensor(-torch.inf).to(probs.device))
    return masked_scores

def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn


class InfoNCEWithAdj(nn.Module):
    def __init__(self, temperature=0.1, normalize=True, chunk_size=512):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.chunk_size = chunk_size

    def _normalize_sparse(self, x):
        if not x.is_sparse:
            return F.normalize(x, p=2, dim=1)

        x_squared = torch.sparse_coo_tensor(
            x._indices(),
            x._values() ** 2,
            x.size()
        )
        row_sums = torch.sparse.sum(x_squared, dim=1).to_dense()
        row_norms = row_sums.sqrt().clamp(min=1e-8)

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
        if self.normalize:
            if hasattr(query, 'is_sparse') and query.is_sparse:
                query = self._normalize_sparse(query)
            else:
                query = F.normalize(query, p=2, dim=1)

            if hasattr(key, 'is_sparse') and key.is_sparse:
                key = self._normalize_sparse(key)
            else:
                key = F.normalize(key, p=2, dim=1)

        if hasattr(query, 'is_sparse') and query.is_sparse:
            query = query.to_dense()
        if hasattr(key, 'is_sparse') and key.is_sparse:
            key = key.to_dense()
        adj = adj.to(query.dtype)

        N = query.size(0)
        chunk_size = min(self.chunk_size, N)  

        global_max_list = []
        global_sum_exp_list = []

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            q_chunk = query[i:end]

            logits_chunk = torch.matmul(q_chunk, key.T) / self.temperature

            current_max = logits_chunk.max(dim=1, keepdim=True).values
            global_max_list.append(current_max)

        global_max = torch.cat(global_max_list, dim=0)

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            q_chunk = query[i:end]

            logits_chunk = torch.matmul(q_chunk, key.T) / self.temperature

            exp_chunk = torch.exp(logits_chunk - global_max[i:end])
            current_sum_exp = exp_chunk.sum(dim=1, keepdim=True)
            global_sum_exp_list.append(current_sum_exp)


        global_sum_exp = torch.cat(global_sum_exp_list, dim=0)
        global_denom = global_max + torch.log(global_sum_exp)

        total_loss = 0.0
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            q_chunk = query[i:end]
            adj_chunk = adj[i:end]

            logits_chunk = torch.matmul(q_chunk, key.T) / self.temperature

            log_softmax_chunk = logits_chunk - global_denom[i:end]

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



def kl_loss(mu, logvar, n_nodes):
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return KLD




