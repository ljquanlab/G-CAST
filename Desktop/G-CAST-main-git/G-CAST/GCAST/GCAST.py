import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
from .utils_func import EarlyStopping
from .Modules import Module
from .loss_fn import  target_distribution, InfoNCEWithAdj, reconstruction_loss,  kl_loss

class GCAST:
    def __init__(self, X, graph_dict,
            lr = 0.05, decay=0.01,
            rec_w=10, kl_w=0.1,  dec_kl_w=1, constrative_w=0.2,
            dec_clsuter_n=10,
            device = 'cuda',
            patience=50,
            temperature=72,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=32,
            gcn_hidden2=16,
            model_path = "", 
            model_name = "checkpoint",
    ):
        self.lr = lr
        self.decay = decay
        self.rec_w = rec_w
        self.kl_w = kl_w
        self.constrative_w = constrative_w
        self.dec_kl_w = dec_kl_w
        self.patience = patience
        self.temperature = temperature
        self.device = device
        self.dec_clsuter_n = dec_clsuter_n
        self.model_name = model_name
        self.cell_num = len(X)

        self.X = torch.Tensor(X.copy()).to(self.device).float()
        self.adj = torch.Tensor(graph_dict["adj"]).to(self.device).float()
        self.adj_prob_feat = torch.Tensor(graph_dict["edge_prob_feat"]).to(self.device).float()
        edge_prob_img = graph_dict.get("edge_prob_img")
        if edge_prob_img is None:
            edge_prob_img = torch.zeros_like(self.adj_prob_feat)
        self.adj_prob_img = torch.Tensor(edge_prob_img).to(self.device).float()
        self.adj_norm = graph_dict["adj_norm"].to(self.device).float()
        self.adj_label = graph_dict["adj_label"].to(self.device).float()
        self.norm_value = graph_dict["norm_value"]

        self.input_dim = self.X.shape[1]
        self.constrative_loss = InfoNCEWithAdj(temperature=self.temperature, normalize=False) 
        self.model = Module(self.input_dim,
                    feat_hidden1=feat_hidden1,feat_hidden2=feat_hidden2,
                    gcn_hidden1=gcn_hidden1,gcn_hidden2=gcn_hidden2,).to(self.device).float()

        self.model_path1  = os.path.join(model_path, f"{model_name}1.pth")
        self.model_path2 = os.path.join(model_path,  f"{model_name}.pth")
        self.early_stopping1 = EarlyStopping(patience=50, verbose=False, path= self.model_path1, mode='save')
        self.early_stopping2 = EarlyStopping(patience=50, verbose=False, path= self.model_path2, mode='save')


    def graph_augment(self, X, adj, drop_edge_rate=0.2, drop_node_rate=0.2):
        if drop_edge_rate == 0 and drop_node_rate==0:
            return X, adj
        num_nodes = X.shape[0]

        # Drop node features
        keep_nodes = torch.rand(num_nodes).to(X.device) > drop_node_rate
        X_aug = X.clone()
        X_aug[~keep_nodes] = 0 

        # Drop edges
        edge_index = adj._indices()
        edge_weight = adj._values()
        mask = torch.rand(edge_weight.shape).to(X.device) > drop_edge_rate
        edge_index_aug = edge_index[:, mask]
        edge_weight_aug = edge_weight[mask]
        adj_aug = torch.sparse_coo_tensor(edge_index_aug, edge_weight_aug, size=adj.shape).coalesce().to(X.device)
        return X_aug, adj_aug


    def train_without_dec(self, epochs=1000, drop_edge_rate=0.2, drop_node_rate=0.2,):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.lr,
            weight_decay=self.decay)

        self.model.train()

        for _ in tqdm(range(epochs)):
            self.optimizer.zero_grad()

            X1, adj1 = self.graph_augment(self.X, self.adj_norm, drop_edge_rate, drop_node_rate)
            X2, adj2 = self.graph_augment(self.X, self.adj_norm, 0, 0)

            z1, mu1, logvar1, de_feat1, = self.model(X1, adj1)
            z2, mu2, logvar2, de_feat2, = self.model(X2, adj2)


            loss_contrastive_img = self.constrative_loss(z1, z2, self.adj_prob_img)*2 + self.constrative_loss(z1, z1, self.adj_prob_img ) +self.constrative_loss(z2, z2, self.adj_prob_img)
            loss_contrastive_feat = self.constrative_loss(z1, z2, self.adj_prob_feat)*2+ self.constrative_loss(z1, z1, self.adj_prob_feat) +self.constrative_loss(z2, z2, self.adj_prob_feat)
            loss_contra = self.constrative_loss(z1, z2, self.adj)*2 + self.constrative_loss(z1, z1, self.adj)
            loss_contrastive = self.norm_value*(loss_contrastive_img + loss_contrastive_feat+loss_contra) /12


            loss_kl1 = kl_loss(mu=mu1,logvar=logvar1, n_nodes=self.cell_num,)
            loss_kl2 = kl_loss(mu=mu2, logvar=logvar2, n_nodes=self.cell_num,)
            loss_kl= (loss_kl1 + loss_kl2)/2

            loss_rec1 = reconstruction_loss(de_feat1, self.X)
            loss_rec2 = reconstruction_loss(de_feat2, self.X)
            loss_rec = (loss_rec1 + loss_rec2)/2

            loss = (
                    self.rec_w * loss_rec +
                    self.constrative_w * loss_contrastive +
                    self.kl_w *loss_kl
            )
            loss.backward()
            self.optimizer.step()
            #
            self.early_stopping1(loss, self.model, self.optimizer, )
            if self.early_stopping1.early_stop:
            #     print("Early stopping triggered.")
                break

   
    def load_model(self, phase="1"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if phase == "1":
            checkpoint = torch.load(self.model_path1, map_location="cpu")
        else:
            checkpoint = torch.load(self.model_path2, map_location="cpu")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        torch.set_rng_state(checkpoint['rng_state'])
        if checkpoint.get('cuda_rng_states') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
  
    def eval_model(self):
        self.model.to(self.device).float().eval()
        with torch.no_grad():
            latent_z, mu, logvar, de_feat = self.model(self.X, self.adj_norm)
            q = self.model.dec(latent_z)

            latent_z = latent_z.detach().cpu().numpy()
            q = q.detach().cpu().numpy()
            out_feat = de_feat.detach().cpu().numpy()
        return latent_z, out_feat, q

    def eval(self):
        self.load_model(phase="2")
        return self.eval_model()[0]

    def train_with_dec(self, epochs=1000, dec_interval=10,):
        self.train_without_dec()
        self.load_model(phase="1")

        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, random_state=2025,n_init=20,  init='k-means++',  max_iter=300  )
        test_z, _, _,  = self.eval_model()
        np.copy(kmeans.fit_predict(test_z))
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, device=self.device).float()
        self.model.train()

        for epoch_id in tqdm(range(epochs)):
            # DEC clustering update
            if epoch_id % dec_interval == 0:
                _,_, tmp_q  = self.eval_model()
                tmp_p = target_distribution(torch.tensor(tmp_q, device=self.device)).float()
                self.model.train()

            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat,= self.model(self.X, self.adj_norm)
            out_q = self.model.dec(latent_z)
            loss_constrative = (self.constrative_loss(latent_z,latent_z,  self.adj_prob_img)
                                + self.constrative_loss(latent_z,latent_z, self.adj)
                                + self.constrative_loss(latent_z,latent_z, self.adj_prob_feat))*self.norm_value /3

            loss_rec = reconstruction_loss(de_feat, self.X)
            # clustering KL loss
            loss_KL1 = kl_loss(mu, logvar, self.cell_num)
            loss_KL2 = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            KL = loss_KL1+loss_KL2
            loss =  (
                    self.dec_kl_w * KL
                     + self.rec_w * loss_rec
                     + self.constrative_w*loss_constrative
                     )

            loss.backward()
            self.optimizer.step()

            self.early_stopping2(loss, self.model, self.optimizer, )
            if self.early_stopping2.early_stop:
            #     print("Early stopping triggered.")
                break

