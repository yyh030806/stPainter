import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
import gc
from typing import Optional, Union, List

def process_sc(
    sc_raw_file_path: str,
    output_path: str
) -> None:
    """Load, QC, filter, and select HVGs for single-cell data."""
    print(f"Loading raw SC data: {sc_raw_file_path}")
    sc_adata = sc.read_h5ad(sc_raw_file_path)
    
    # Calculate QC metrics
    sc_adata.var['mt'] = sc_adata.var_names.str.startswith('MT-')
    sc_adata.var['ribo'] = sc_adata.var_names.str.startswith(('RPS', 'RPL'))
    sc.pp.calculate_qc_metrics(sc_adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True) 
    
    # Filter cells
    n_obs_before = sc_adata.n_obs
    min_genes_per_cell = 300
    min_counts_per_cell = 500
    num_top_genes = 10000
    max_mito_pct = 20
    max_ribo_pct = 40
    
    sc_adata = sc_adata[
        (sc_adata.obs['n_genes_by_counts'] > min_genes_per_cell) &
        (sc_adata.obs['pct_counts_mt'] < max_mito_pct) &
        (sc_adata.obs['total_counts'] > min_counts_per_cell) &
        (sc_adata.obs['pct_counts_ribo'] < max_ribo_pct)
    ].copy()
    print(f"Cells removed via QC: {n_obs_before - sc_adata.n_obs}")
    
    # Filter genes
    min_cells = 3 
    n_vars_before = sc_adata.n_vars
    sc.pp.filter_genes(sc_adata, min_cells=min_cells)
    print(f"Genes removed: {n_vars_before - sc_adata.n_vars}")
    
    # HVG selection
    sc_adata.raw = sc_adata.copy()
        
    sc_adata.layers['counts'] = sc_adata.X.copy()
    
    batch_key = "batch" if "batch" in sc_adata.obs.columns else None
    
    sc.pp.highly_variable_genes(
        sc_adata,
        n_top_genes=num_top_genes,
        flavor="seurat_v3",
        layer="counts",
        subset=True,
        batch_key=batch_key,
    )
    
    hvg_genes = sc_adata.var_names[sc_adata.var['highly_variable']]
    
    # Save processed data
    sc_adata = sc_adata[:, hvg_genes].copy()
    sc_adata.write(output_path)
    print(f"Saved processed SC data: {sc_adata.shape}")
    
    del sc_adata
    gc.collect()

def calculate_gene_sparsity(
    sc_file_path: str,
    output_path: str,
    group_key: str = 'Tumor Type'
) -> None:
    """Compute gene sparsity (1 - density) per group."""
    print(f"Calculating sparsity: {sc_file_path}")
    adata = sc.read_h5ad(sc_file_path)
    
    if group_key not in adata.obs.columns:
        print(f"Missing column '{group_key}', skipping sparsity.")
        return

    sparsity_dict = {}
    unique_types = adata.obs[group_key].unique()

    for t_type in unique_types:
        mask = (adata.obs[group_key] == t_type).values
        subset_X = adata.X[mask]
        n_cells = subset_X.shape[0]
        
        if n_cells == 0:
            continue

        if scipy.sparse.issparse(subset_X):
            non_zeros = subset_X.getnnz(axis=0)
        else:
            non_zeros = np.count_nonzero(subset_X, axis=0)
        
        if isinstance(non_zeros, np.matrix):
            non_zeros = np.array(non_zeros).flatten()
            
        sparsity_dict[t_type] = 1.0 - (non_zeros / n_cells)
        del subset_X

    df = pd.DataFrame(sparsity_dict, index=adata.var_names)
    df.to_csv(output_path)    
    print(f"Sparsity ratio saved: {output_path}")
    
    del adata, df, sparsity_dict
    gc.collect()

def process_st(sc_ref_path: str, st_raw_file_path: str, output_path: str) -> None:
    """Align ST data to SC feature space and filter."""
    # Load SC reference genes
    sc_adata = sc.read_h5ad(sc_ref_path, backed='r')
    sc_genes = sc_adata.var_names.copy()
    n_sc_vars = sc_adata.n_vars
    del sc_adata
    gc.collect()
    
    st_adata = sc.read_h5ad(st_raw_file_path)
    st_original_genes = st_adata.var_names

    # Align genes
    common_genes, st_idx, sc_idx = np.intersect1d(st_original_genes, sc_genes, return_indices=True)

    aligned_st_X = scipy.sparse.lil_matrix((st_adata.n_obs, n_sc_vars), dtype=st_adata.X.dtype)
    aligned_st_X[:, sc_idx] = st_adata.X[:, st_idx]
    aligned_st_X = aligned_st_X.tocsr()

    gene_mask = sc_genes.isin(st_original_genes).astype(int)
    aligned_var = pd.DataFrame(index=sc_genes)
    aligned_var['impute_mask'] = gene_mask
    
    # Copy metadata
    obs_copy = st_adata.obs.copy()
    obsm_spatial = st_adata.obsm['spatial'].copy() if 'spatial' in st_adata.obsm else None
    uns_spatial = st_adata.uns['spatial'].copy() if 'spatial' in st_adata.uns else None
    
    del st_adata
    gc.collect()

    aligned_st_adata = sc.AnnData(X=aligned_st_X, obs=obs_copy, var=aligned_var)
    if obsm_spatial is not None: aligned_st_adata.obsm['spatial'] = obsm_spatial
    if uns_spatial is not None: aligned_st_adata.uns['spatial'] = uns_spatial
    
    del aligned_st_X, obs_copy, obsm_spatial
    gc.collect()
    
    # Remove empty spots
    n_spots_before = aligned_st_adata.n_obs
    spots_to_keep = np.asarray(aligned_st_adata.X.sum(axis=1)).flatten() > 0   
    aligned_st_adata = aligned_st_adata[spots_to_keep, :].copy()
    print(f"Empty spots removed: {n_spots_before - aligned_st_adata.n_obs}")
    
    # Filter low count spots (bottom 15%)
    total_counts = np.array(aligned_st_adata.X.sum(axis=1)).flatten()
    cutoff = np.quantile(total_counts, 0.15)
    spots_to_keep = total_counts >= cutoff
    n_before_filter = aligned_st_adata.n_obs
    aligned_st_adata = aligned_st_adata[spots_to_keep, :].copy()
    print(f"Low quality spots removed: {n_before_filter - aligned_st_adata.n_obs}")
    
    print(f"Aligned ST data shape: {aligned_st_adata.shape}")
    aligned_st_adata.write(output_path)
    
    del aligned_st_adata, spots_to_keep
    gc.collect()