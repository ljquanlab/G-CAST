import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial


def sce_loss(x, y, alpha=3):
    #
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
# nn.PReLU()
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight) # 稠密矩阵之间的乘法
        output = torch.spmm(adj, support) # 左稀疏 右稠密
        output = self.act(output)
        return output


class MultiGraphConvolution(nn.Module):
    """
    Multi-layer Graph Convolution Network
    Args:
        layer_dims (list): List of layer dimensions [input_dim, hidden_dim1, ..., output_dim]
        dropout (float): Dropout rate applied to each layer
        activations (list): Activation functions for each layer (length = len(layer_dims)-1)
    """

    def __init__(self, layer_dims, dropout=0., activations=None):
        super(MultiGraphConvolution, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(layer_dims) - 1

        # Default activations: ReLU for hidden layers, identity for last layer
        if activations is None:
            activations = [F.relu] * (self.num_layers - 1) + [lambda x: x]

        assert len(activations) == self.num_layers, \
            "Number of activations must match number of layers"

        # Create GCN layers
        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            act = activations[i]
            self.layers.append(
                GraphConvolution(in_dim, out_dim, dropout, act)
            )

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mask):
        col = mask.coalesce().indices()[0]
        row = mask.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))

        return result


class Module(nn.Module):
    def __init__(
            self,
            input_dim,
            layer_num=1,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=32,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
    ):
        super(Module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2
        self.layer_num = layer_num

        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        if self.layer_num == 1:
            self.decoder = GraphConvolution(self.latent_dim, self.input_dim,  self.p_drop, act=lambda x: x)
        elif self.layer_num > 1:
            layer_dim = [self.latent_dim] + [self.input_dim] * self.layer_num
            self.decoder = MultiGraphConvolution(
                layer_dims=layer_dim,
                dropout=self.p_drop,
                activations=[F.relu] * (self.layer_num - 1) + [lambda x: x],
            )
        if self.layer_num == 1:
            self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        elif self.layer_num > 1:
            layer_dim = [self.feat_hidden2] + [self.gcn_hidden1]*self.layer_num
            self.gc1 = MultiGraphConvolution(
                layer_dims=layer_dim,
                dropout=self.p_drop,
                activations=[F.relu] * self.layer_num ,
            )
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)

        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)

        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.gcn_hidden2 + self.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse": # meas square error
            criterion = nn.MSELoss()
        elif loss_fn == "sce": # Symmetric Cross Entropy
            criterion = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return criterion

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z, adj)

        return z, mu, logvar, de_feat

    def dec(self, z):
        # DEC clustering z.shape (num_cell, gcn2_hidden+feat_x.dim)
        # print(self.cluster_layer.shape) [10, 32]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t() 
        return q
