import os
import torch
import numpy as np
import scanpy as sc
import random



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
                 mode: str = 'save'): 

        assert mode in ('save', 'return')
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.mode = mode

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None  

    def __call__(self, val_loss, model, optimizer=None, epoch=None):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            if self.verbose:
                if self.best_loss is None:
                    print(f"Initial validation loss set to {val_loss:.6f}. Saving checkpoint...")
                else:
                    print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving checkpoint...")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'epoch': epoch if epoch is not None else -1,
                'best_loss': val_loss
            }


            if self.mode == 'save':
                torch.save(checkpoint, self.path)
                if self.verbose:
                    print(f"Checkpoint saved to {self.path}")
            elif self.mode == 'return':
                self.best_state = {k: v for k, v in checkpoint.items()}
                if self.verbose:
                    print("Checkpoint cached in memory.")

            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
