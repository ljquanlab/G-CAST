import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt


def plot_spatial(adata, color, title, save=False, show=True):
    sc.pl.spatial(adata,
                  color=color,
                  title=title,
                  spot_size=180,
                  show=show                 )
    if save:
        save_path = f"{title}.png"
        plt.savefig(save_path)
    plt.close()



def plot_umap(adata, color, title, use_rep="emb", metric='cosine', save=False, show=True, random_state=2025):
    sc.pp.neighbors(adata, use_rep=use_rep, metric=metric, random_state=2025)
    sc.tl.umap(adata)
    sc.pl.umap(adata,
               color=color,
               show=show
               )
    if save:
        save_path = f"{title}.png"
        plt.savefig(save_path)
    plt.close()
    
    