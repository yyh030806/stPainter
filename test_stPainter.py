import os
import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from scvi.module import VAE
from src.models import GiT
from src.modules import DiffusionModule, VAEModule
from src.data import STDataset

from src.parsing import *
from src.metrics import CalculateMetrics

def test_gene_metrics(diffusion_module, test_loader, true_indices, impute_mask, args, X_ground_truth, gene_names):
    """Evaluate gene imputation performance using K-Fold cross-validation."""
    print(f"\n[Evaluation] Starting {args.n_splits}-fold cross-validation for Gene Metrics...")
    
    X_imputed_all_folds = np.zeros_like(X_ground_truth, dtype=np.float32)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    for i, (train_idx_rel, test_idx_rel) in enumerate(kf.split(true_indices)):
        print(f"  Processing Fold {i+1}/{args.n_splits}...")
        
        train_indices = true_indices[train_idx_rel]
        test_indices = true_indices[test_idx_rel]
        
        # Mask setup
        known_mask = torch.zeros_like(impute_mask, dtype=torch.bool, device=impute_mask.device)
        known_mask[train_indices] = True
        
        unknown_mask = torch.zeros_like(impute_mask, dtype=torch.bool, device=impute_mask.device)
        unknown_mask[test_indices] = True

        # Inference
        X_imputed_this_fold = diffusion_module.impute(test_loader, known_mask, use_sparity_ratio=True, return_latent=False)

        # Aggregate results
        unknown_mask_np = unknown_mask.squeeze().cpu().numpy()
        X_imputed_all_folds[:, unknown_mask_np] = X_imputed_this_fold[:, unknown_mask_np]
    
    # Save sparse h5ad
    save_dir = os.path.join(args.result_dir, args.cancer_type)
    os.makedirs(save_dir, exist_ok=True)
    
    adata = ad.AnnData(X=sparse.csr_matrix(X_imputed_all_folds))
    adata.var_names = gene_names
    save_path = os.path.join(save_dir, 'stPainter_gene.h5ad')
    adata.write_h5ad(save_path)
    print(f"  Saved results to {save_path}")

    # Metrics calculation
    print("  Calculating Gene Metrics...")
    impute_mask_np = impute_mask.squeeze().cpu().numpy()
    
    raw_count = pd.DataFrame(X_ground_truth[:, impute_mask_np], columns=gene_names[impute_mask_np])
    recover_count_sample = pd.DataFrame(X_imputed_all_folds[:, impute_mask_np], columns=gene_names[impute_mask_np])
    
    metric_calculator = CalculateMetrics(raw_count, recover_count_sample, logp1=True)
    metric_calculator.run_gene_metrics(save_dir, prefix='stPainter')


def test_cluster_metrics(diffusion_module, test_loader, impute_mask, args, X_ground_truth, gene_names, labels):
    """Evaluate clustering performance using all known genes for imputation."""
    print(f"\n[Evaluation] Starting Full Imputation for Clustering Metrics...")
    
    known_mask = impute_mask.clone()
    
    # Inference
    X_imputed, Z_latent = diffusion_module.impute(test_loader, known_mask, use_sparity_ratio=False, return_latent=True)
    
    # Save gene h5ad
    save_dir = os.path.join(args.result_dir, args.cancer_type)
    os.makedirs(save_dir, exist_ok=True)

    adata = ad.AnnData(X=sparse.csr_matrix(X_imputed))
    adata.var_names = gene_names
    if labels is not None:
        adata.obs['label'] = labels
        adata.obs['label'] = adata.obs['label'].astype('category')

    save_path = os.path.join(save_dir, 'stPainter_cluster_gene.h5ad')
    adata.write_h5ad(save_path)
    print(f"  Saved gene results to {save_path}")
    
    # save latent h5ad
    adata = ad.AnnData(X=Z_latent)
    if labels is not None:
        adata.obs['label'] = labels
        adata.obs['label'] = adata.obs['label'].astype('category')
    save_path = os.path.join(save_dir, 'stPainter_cluster_latent.h5ad')
    adata.write_h5ad(save_path)
    print(f"  Saved latent results to {save_path}")
    
    # Metrics calculation
    print("  Calculating Cluster Metrics...")
    raw_df = pd.DataFrame(X_ground_truth, columns=gene_names)
    imputed_df = pd.DataFrame(X_imputed, columns=gene_names)
    latent_df = pd.DataFrame(Z_latent)
    
    metric_calculator = CalculateMetrics(raw_df, imputed_df, latent=latent_df, logp1=True)
    
    if labels is not None:
        metric_calculator.run_cluster_metrics(save_dir, 'stPainter', labels=labels, use_rep='latent')
    else:
        print("  [Warning] No labels found, skipping metrics.")

def main(args):
    pl.seed_everything(42, workers=True)
    
    # Init models
    vae_model = VAE(
        n_input=args.input_size,
        n_batch=args.num_classes,
        n_hidden=args.hidden_size_vae,
        n_latent=args.latent_size,
        n_layers=args.num_layer,
    )
    
    sit_model = GiT(
        latent_size=args.latent_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size_sit,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        class_dropout_prob=args.class_dropout_prob,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
    )
    
    # Load checkpoints
    device = f"cuda:{args.devices[0]}"
        
    if not args.vae_checkpoint:
        raise ValueError("VAE checkpoint required.")
    vae_module = VAEModule.load_from_checkpoint(checkpoint_path=args.vae_checkpoint, 
                                                args=args, model=vae_model, map_location=device)
    
    if not args.diffusion_checkpoint:
        raise ValueError("Diffusion checkpoint required.")
    diffusion_module = DiffusionModule.load_from_checkpoint(checkpoint_path=args.diffusion_checkpoint, 
                                                            args=args, model=sit_model, vae=vae_module, map_location=device)
    
    # Load data
    test_dataset = STDataset('st', args.input_path, args.cancer_type)
    print(f"Cells: {test_dataset.get_num_cell()}, Genes: {test_dataset.get_num_gene()}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_cpus)

    # Setup masks and GT
    impute_mask = test_dataset.get_impute_mask().to(device)
    true_indices = impute_mask.squeeze().nonzero(as_tuple=True)[0]
    
    X_ground_truth = np.concatenate(
        [batch[0].cpu().detach().numpy() for batch in test_loader], 
        axis=0
    )

    labels = test_dataset.get_annotation()
    _, gene_names = test_dataset.get_names()
    
    # Run evaluations
    test_gene_metrics(diffusion_module, test_loader, true_indices, impute_mask, args, X_ground_truth, gene_names)
    
    test_cluster_metrics(diffusion_module, test_loader, impute_mask, args, X_ground_truth, gene_names, labels)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parse_testing_impute_args(parser)
    parse_model_args(parser)
    parse_checkpoint_args(parser)
    parse_transport_args(parser)
    parse_sampling_args(parser)
    parse_data_args(parser)
    
    temp_args, _ = parser.parse_known_args()
    if temp_args.mode == 'ODE':
        parse_ode_args(parser)
    elif temp_args.mode == 'SDE':
        parse_sde_args(parser)

    args = parser.parse_args()
    main(args)