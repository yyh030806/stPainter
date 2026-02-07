import scanpy as sc
import numpy as np
import torch
import scipy.sparse
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Union

TUMOR_TO_IDX = {
    'Unknown': 0, 
    'OV': 1, 
    'UVM': 2, 
    'SCLC': 3, 
    'TGCT': 4, 
    'GBM': 5, 
    'COAD': 6, 
    'KIRC': 7, 
    'THYM': 8, 
    'STAD': 9, 
    'THCA': 10, 
    'ESCC': 11, 
    'NSCLC': 12, 
    'PAAD': 13, 
    'LAML': 14, 
    'PRAD': 15, 
    'LIHC': 16, 
    'BRCA': 17, 
    'BLCA': 18, 
    'CESC': 19, 
    'SARC': 20
}

class SCDataset(Dataset):
    def __init__(self, name: str, sc_data_file_path: str):
        self.name = name
        self.sc_adata = sc.read_h5ad(sc_data_file_path)

        # Handle sparse matrices
        if scipy.sparse.issparse(self.sc_adata.X):
            self.X = self.sc_adata.X.toarray()
        else:
            self.X = self.sc_adata.X
        
        # Map tumor types
        if 'Tumor Type' in self.sc_adata.obs.columns:
            self.tumor_type = self.sc_adata.obs['Tumor Type'].tolist()
            # Map unknown types to 0 (Unknown)
            self.tumor_idx = [TUMOR_TO_IDX.get(t, 0) for t in self.tumor_type]
        else:
            self.tumor_idx = [0] * self.X.shape[0]
            
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx], dtype=torch.int), torch.tensor(self.tumor_idx[idx], dtype=torch.int)
    
    def get_num_cell(self) -> int:
        return self.sc_adata.shape[0]
        
    def get_num_gene(self) -> int:
        return self.sc_adata.shape[1]
    
    def get_names(self) -> Tuple[list, list]:
        return self.sc_adata.obs_names, self.sc_adata.var_names
    

class STDataset(Dataset):
    def __init__(self, name: str, st_data_file_path: str, tumor_type: str = 'Unknown'):
        self.name = name
        self.st_adata = sc.read_h5ad(st_data_file_path)

        # Handle sparse matrices
        if scipy.sparse.issparse(self.st_adata.X):
            self.X = self.st_adata.X.toarray()
        else:
            self.X = self.st_adata.X
        
        # Map tumor types
        if 'Tumor Type' in self.st_adata.obs_keys():
            self.tumor_type = self.st_adata.obs['Tumor Type'].tolist()
            self.tumor_idx = [TUMOR_TO_IDX.get(t, 0) for t in self.tumor_type]
        else: 
            # Use provided argument, default to 0 if invalid
            idx = TUMOR_TO_IDX.get(tumor_type, 0)
            self.tumor_idx = [idx] * self.get_num_cell()
                        
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx], dtype=torch.int), torch.tensor(self.tumor_idx[idx], dtype=torch.int)
    
    def get_num_cell(self) -> int:
        return self.st_adata.shape[0]
        
    def get_num_gene(self) -> int:
        return self.st_adata.shape[1]
    
    def get_names(self) -> Tuple[list, list]:
        return self.st_adata.obs_names, self.st_adata.var_names
    
    def get_impute_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.st_adata.var['impute_mask'].to_numpy()).to(dtype=torch.bool)
    
    def get_annotation(self) -> List:
        if 'annotation' in self.st_adata.obs.columns:
            return self.st_adata.obs['annotation'].tolist()
        return []
    
    def save_imputed(self, X: Union[np.ndarray, torch.Tensor]):
        if not isinstance(X, np.ndarray):
            X = X.detach().cpu().numpy() if hasattr(X, 'detach') else X
        
        if X.shape[0] != self.st_adata.shape[0]:
            raise ValueError(f"Shape mismatch: X {X.shape} vs AnnData {self.st_adata.shape}")

        self.st_adata.layers['imputed'] = scipy.sparse.csr_matrix(X)
        
    def save_latent(self, X: Union[np.ndarray, torch.Tensor]):
        if not isinstance(X, np.ndarray):
            X = X.detach().cpu().numpy() if hasattr(X, 'detach') else X
        
        if X.shape[0] != self.st_adata.shape[0]:
            raise ValueError(f"Shape mismatch: X {X.shape} vs AnnData {self.st_adata.shape}")

        self.st_adata.obsm['latent'] = scipy.sparse.csr_matrix(X)
            
    def store_result(self, output_file_path: Optional[str] = None):
        if output_file_path:
            self.st_adata.write(output_file_path)