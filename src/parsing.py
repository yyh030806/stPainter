import argparse
from typing import Optional

def none_or_str(value: str) -> Optional[str]:
    """Helper for argparse to accept 'None' string and convert to Python None."""
    if not value or value.lower() == 'none':
        return None
    return value

# ==========================================
# Mode & Data Arguments
# ==========================================

def parse_data_args(parser: argparse.ArgumentParser):
    """Arguments for data paths and directories."""
    group = parser.add_argument_group("Data Path Arguments")
    group.add_argument("--data_path", type=str, default="./data/processed/sc_processed.h5ad", 
                       help="Path to the processed .h5ad data file.")
    group.add_argument('--raw_data_dir', type=str, default="./data/raw", 
                       help="Directory containing raw input files")
    group.add_argument('--processed_data_dir', type=str, default="./data/processed", 
                       help="Directory for output files")

# ==========================================
# Model Architecture Arguments
# ==========================================

def parse_model_args(parser: argparse.ArgumentParser):
    """Arguments for VAE and Diffusion model architecture."""
    group = parser.add_argument_group("Model Architecture Arguments")
    
    # General Dimensions
    group.add_argument("--input_size", type=int, default=10000, help="Input feature dimension.")
    group.add_argument("--latent_size", type=int, default=50, help="Dimension of the latent space.")
    group.add_argument('--num_classes', type=int, default=21, help='Number of conditional classes.')

    # VAE Specific
    group.add_argument("--hidden_size_vae", type=int, default=256, help="Hidden layer size for VAE.")
    group.add_argument("--num_layer", type=int, default=4, help="Number of hidden layers in VAE.")
    
    # Diffusion Specific
    group.add_argument('--patch_size', type=int, default=1, help='Patch size (1 for non-image data).')
    group.add_argument('--hidden_size_sit', type=int, default=128, help='Transformer hidden dimension.')
    group.add_argument('--depth', type=int, default=8, help='Number of transformer layers.')
    group.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    group.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP hidden/embedding ratio.')
    group.add_argument('--class_dropout_prob', type=float, default=0.1, help='Classifier dropout probability.')
    group.add_argument('--learn_sigma', action='store_true', help='Whether to learn the diffusion model sigma.')

def parse_checkpoint_args(parser: argparse.ArgumentParser):
    """Arguments for loading pretrained checkpoints."""
    group = parser.add_argument_group("Checkpoint Arguments")
    group.add_argument('--vae_checkpoint', type=none_or_str, default="checkpoint/vae.ckpt", help='Path to pretrained VAE checkpoint.')
    group.add_argument('--diffusion_checkpoint', type=none_or_str, default="checkpoint/diffusion.ckpt", help='Path to pretrained Diffusion checkpoint.')

# ==========================================
# Training Arguments (VAE & Diffusion)
# ==========================================

def parse_training_vae_args(parser: argparse.ArgumentParser):
    """Arguments specifically for VAE training."""
    group = parser.add_argument_group("Training VAE Arguments")
    group.add_argument("--epochs", type=int, default=200, help="Total training epochs.")
    group.add_argument("--batch_size", type=int, default=1024, help="Training batch size.")
    group.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    
    group.add_argument("--devices", type=int, nargs='+', default=[0], help="GPU device IDs.")
    group.add_argument("--num_cpus", type=int, default=4, help="Num CPU workers.")
    
    group.add_argument("--eval_epoch", type=int, default=20, help="Validate every N epochs.")
    group.add_argument("--save_epoch", type=int, default=20, help="Save checkpoint every N epochs.")
    group.add_argument("--log_every_n_steps", type=int, default=20, help="Log frequency within epoch.")

def parse_training_diffusion_args(parser: argparse.ArgumentParser):
    """Arguments specifically for Diffusion training."""
    group = parser.add_argument_group("Training Diffusion Arguments")
    group.add_argument('--epochs', type=int, default=1000)
    group.add_argument('--batch_size', type=int, default=1024)
    group.add_argument('--lr', type=float, default=1e-4, help='AdamW learning rate.')
    
    group.add_argument('--devices', type=int, nargs='+', default=[0], help='GPU device IDs.')
    group.add_argument('--num_cpus', type=int, default=8)
    
    group.add_argument("--eval_epoch", type=int, default=100, help="Validate every N epochs.")
    group.add_argument("--save_epoch", type=int, default=100, help="Save checkpoint every N epochs.")
    group.add_argument("--log_every_n_steps", type=int, default=20, help="Log frequency.")
    group.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate.')

# ==========================================
# Testing & Imputation Arguments
# ==========================================

def parse_testing_impute_args(parser: argparse.ArgumentParser):
    """Arguments for testing and imputation tasks."""
    group = parser.add_argument_group("Testing Impute Arguments")
    group.add_argument('--devices', type=int, nargs='+', default=[0], help='GPU device IDs.')
    group.add_argument('--num_cpus', type=int, default=8)
    group.add_argument('--batch_size', type=int, default=1024)
    group.add_argument('--t_forward', type=float, default=0.9, help='Forward diffusion noise level (t).')
    
    group.add_argument('--input_path', type=str, default=None, help='Input data path.')
    group.add_argument('--result_dir', type=str, default='./result/stPainter', help='Directory for results.')
    group.add_argument('--output_path', type=str, default=None, help='Output data path.')
    group.add_argument('--cancer_type', type=str, default='Unknown', help='Cancer type context.')
    
    group.add_argument('--n_splits', type=int, default=5, help='Cross-validation splits.')
    group.add_argument('--gene_sparsity_ratio_file_path', type=str, default='./data/processed/gene_sparsity_ratio.csv', help='Path to sparsity file.')

# ==========================================
# Sampling & Transport Arguments
# ==========================================

def parse_transport_args(parser: argparse.ArgumentParser):
    """Arguments for the transport map / noise schedule."""
    group = parser.add_argument_group("Transport Arguments")
    group.add_argument("--path_type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss_weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample_eps", type=float, default=None)
    group.add_argument("--train_eps", type=float, default=None)

def parse_sampling_args(parser: argparse.ArgumentParser):
    """General arguments for sampling (ODE/SDE shared)."""
    group = parser.add_argument_group("General Sampling Arguments")
    group.add_argument('--mode', type=str, default='ODE', choices=['ODE', 'SDE'], help='Sampling mode.')
    group.add_argument('--cfg_scale', type=float, default=1.0, help='Classifier-Free Guidance scale.')
    group.add_argument('--num_steps', type=int, default=30, help='Number of sampling steps.')

def parse_ode_args(parser: argparse.ArgumentParser):
    """Arguments specifically for ODE solvers."""
    group = parser.add_argument_group("ODE Sampler Arguments")
    group.add_argument("--sampling_method", type=str, default="euler", choices=['dopri5', 'euler', 'midpoint'], help="Solver method.")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance.")
    group.add_argument("--reverse", action="store_true", help="Reverse time flow.")
    group.add_argument("--likelihood", action="store_true", help="Enable likelihood computation.")

def parse_sde_args(parser: argparse.ArgumentParser):
    """Arguments specifically for SDE solvers."""
    group = parser.add_argument_group("SDE Sampler Arguments")
    group.add_argument("--sampling_method", type=str, default="Euler", choices=["Euler", "Heun"], help="Solver method.")
    group.add_argument("--diffusion_form", type=str, default="sigma", 
                       choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing_decreasing"], 
                       help="Diffusion coefficient form.")
    group.add_argument("--diffusion_norm", type=float, default=1.0)
    group.add_argument("--last_step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"], help="Last step correction strategy.")
    group.add_argument("--last_step_size", type=float, default=0.04)