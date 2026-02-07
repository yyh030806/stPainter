import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    normalized_mutual_info_score
)
from typing import Optional, List, Dict, Union

# ==============================
# Helper Functions
# ==============================

def cal_ssim(im1: np.ndarray, im2: np.ndarray, M: float) -> float:
    mu1, mu2 = im1.mean(), im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    
    k1, k2, L = 0.01, 0.03, M
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    C3 = C2 / 2
    
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return l12 * c12 * s12

def scale_max(df):
    max_vals = df.max()
    max_vals[max_vals == 0] = 1 
    return df / max_vals

def scale_z_score(df):
    return pd.DataFrame(
        st.zscore(df, nan_policy='omit'), 
        index=df.index, columns=df.columns
    ).fillna(0)

def scale_plus(df):
    df = df.clip(lower=0)
    sum_vals = df.sum()
    sum_vals[sum_vals == 0] = 1
    return df / sum_vals

def get_clustering_scores(labels_true, labels_pred) -> Dict[str, float]:
    return {
        "ARI": adjusted_rand_score(labels_true, labels_pred),
        "AMI": adjusted_mutual_info_score(labels_true, labels_pred),
        "Homo": homogeneity_score(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred)
    }

# ==============================
# Main Metrics Class
# ==============================

class CalculateMetrics:
    def __init__(self, raw_count, impute_count=None, latent=None, logp1=False):
        # Align indices
        if impute_count is not None:
            impute_count.index = list(raw_count.index)
        
        self.raw_count = self._standardize(raw_count)
        self.impute_count = self._standardize(impute_count)
        self.latent = latent 
        
        self.is_log_transformed = False
        if logp1 and impute_count is not None:
            self.raw_count = np.log1p(self.raw_count)
            self.impute_count = np.log1p(self.impute_count)
            self.is_log_transformed = True

    def _standardize(self, df):
        if not isinstance(df, pd.DataFrame): return df
        # Format columns and remove duplicates
        df.columns = [str(x).upper() for x in df.columns]
        df = df.loc[~df.index.duplicated(keep='first')]
        return df.fillna(1e-20)
    
    def get_hvg_list(self):
        adata = sc.AnnData(self.raw_count)
        if not self.is_log_transformed:
            sc.pp.log1p(adata)
            
        try:
            sc.pp.highly_variable_genes(adata, flavor='seurat', inplace=True)
            col = 'dispersions_norm' if 'dispersions_norm' in adata.var.columns else 'dispersions'
            hvg_ranked = adata.var.sort_values(by=col, ascending=False).index
        except Exception:
            # Fallback to simple variance
            adata.var['variance'] = adata.X.var(axis=0)
            hvg_ranked = adata.var.sort_values(by='variance', ascending=False).index
            
        return list(hvg_ranked)

    # --- Metrics Calculation (Optimized using dicts) ---

    def SSIM(self, raw, impute):
        raw, impute = scale_max(raw), scale_max(impute)
        res = {}
        for label in raw.columns:
            r, i = raw[label].fillna(1e-20), impute[label].fillna(1e-20)
            M = max(r.max(), i.max())
            val = 1.0 if M == 0 else cal_ssim(r.values.reshape(-1, 1), i.values.reshape(-1, 1), M)
            res[label] = val
        return pd.DataFrame(res, index=["SSIM"])

    def PCC(self, raw, impute):
        res = {}
        for label in raw.columns:
            r, i = raw[label].fillna(1e-20), impute[label].fillna(1e-20)
            val = st.pearsonr(r, i)[0] if (r.std() > 0 and i.std() > 0) else 0.0
            res[label] = val
        return pd.DataFrame(res, index=["PCC"])

    def JS(self, raw, impute):
        raw, impute = scale_plus(raw), scale_plus(impute)
        res = {}
        for label in raw.columns:
            r, i = raw[label].fillna(1e-20), impute[label].fillna(1e-20)
            M = (r + i) / 2
            val = 0.5 * st.entropy(r, M) + 0.5 * st.entropy(i, M)
            res[label] = val
        return pd.DataFrame(res, index=["JS"])

    def RMSE(self, raw, impute):
        raw, impute = scale_z_score(raw), scale_z_score(impute)
        res = {}
        for label in raw.columns:
            r, i = raw[label].fillna(1e-20), impute[label].fillna(1e-20)
            val = np.sqrt(((r - i) ** 2).mean())
            res[label] = val
        return pd.DataFrame(res, index=["RMSE"])
    
    # --- Clustering Logic ---

    def _cluster_and_score(self, adata, labels, mode='gene'):
        adata.obs['ground_truth'] = labels

        # Preprocessing differs by mode
        if mode == 'gene':
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, n_pcs=30, n_neighbors=20)
        else: # latent
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=20)
        
        # Cluster
        res = 0.02
        if 'leiden' in dir(sc.tl):
            sc.tl.leiden(adata, key_added='pred', resolution=res)
        else:
            sc.tl.louvain(adata, key_added='pred', resolution=res)
        

        print(adata.obs['ground_truth'])
        print(adata.obs['pred'])
        scores = get_clustering_scores(adata.obs['ground_truth'], adata.obs['pred'])
        return pd.DataFrame([scores])

    def _get_subset(self, n_genes):
        full_hvg = self.get_hvg_list()
        top_k = full_hvg[:min(n_genes, len(full_hvg))]
        return self.raw_count[top_k], self.impute_count[top_k]

    # --- Runners ---

    def run_gene_metrics(self, output_dir='.', prefix='', n_genes=300):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        raw_sub, impute_sub = self._get_subset(n_genes)
        
        # Calculate all metrics
        df = pd.concat([
            self.PCC(raw_sub, impute_sub).T,
            self.SSIM(raw_sub, impute_sub).T,
            self.RMSE(raw_sub, impute_sub).T,
            self.JS(raw_sub, impute_sub).T
        ], axis=1)
        
        # Print summary
        print(f"\n>>> Average Metrics for Top HVGs (Prefix: {prefix}) <<<")
        print(f"{'Top N':<10} {'PCC':<10} {'SSIM':<10} {'RMSE':<10} {'JS':<10}")
        print("-" * 55)
        for k in [10, 50, 100, 200, 300]:
            if k <= len(df):
                m = df.iloc[:k].mean()
                print(f"{k:<10} {m['PCC']:<10.4f} {m['SSIM']:<10.4f} {m['RMSE']:<10.4f} {m['JS']:<10.4f}")
        print("-" * 55 + "\n")
        
        df.to_csv(os.path.join(output_dir, f"{prefix}_gene_metrics.csv"), float_format='%.8f')
        return df

    def run_cluster_metrics(self, output_dir='.', prefix='', labels=None, n_genes=10000, use_rep='gene'):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if labels is None: return pd.DataFrame() # Error handle

        try:
            if use_rep == 'gene':
                print(f"Clustering imputed expression (Top {n_genes} HVGs)...")
                _, impute_sub = self._get_subset(n_genes)
                if len(labels) != len(impute_sub): raise ValueError("Label mismatch")
                df = self._cluster_and_score(sc.AnnData(impute_sub), labels, mode='gene')
                
            elif use_rep == 'latent':
                print("Clustering latent space...")
                if self.latent is None: raise ValueError("No latent data")
                if len(labels) != len(self.latent): raise ValueError("Label mismatch")
                df = self._cluster_and_score(sc.AnnData(self.latent), labels, mode='latent')
            else:
                raise ValueError("Invalid use_rep")

        except Exception as e:
            print(f"Clustering error ({use_rep}): {e}")
            df = pd.DataFrame({"ARI": [0], "AMI": [0], "Homo": [0], "NMI": [0]})

        df.to_csv(os.path.join(output_dir, f"{prefix}_cluster_metrics.csv"), index=False, float_format='%.8f')
        return df