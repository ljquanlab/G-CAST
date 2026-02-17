# import dgl
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from scipy.sparse import coo_matrix
from torch.autograd import Variable
from sklearn import metrics


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            if self.verbose:
                if self.best_loss is None:
                    print(f"Initial validation loss set to {val_loss:.6f}. Saving model...")
                else:
                    print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model...")
                # print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model...")
            torch.save(model.state_dict(), self.path)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True