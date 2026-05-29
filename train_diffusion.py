import os
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

# Models
from scvi.module import VAE
from src.models import GiT
from src.modules import DiffusionModule, VAEModule
from src.data import SCDataset

# Argument Parsing
from src.parsing import (
    parse_training_diffusion_args,
    parse_model_args,
    parse_checkpoint_args,
    parse_transport_args,
    parse_sampling_args,
    parse_data_args,
    parse_ode_args,
    parse_sde_args
)

def main(args):
    pl.seed_everything(42, workers=True)
    device = f"cuda:{args.devices[0]}" if torch.cuda.is_available() else "cpu"
    
    print(f">>> Task: Training Diffusion Model")
    
    # Initialize VAE architecture
    vae_model = VAE(
        n_input=args.input_size,
        n_batch=args.num_classes,
        n_hidden=args.hidden_size_vae,
        n_latent=args.latent_size,
        n_layers=args.num_layer,
    )
    
    # Load pretrained VAE weights
    if not args.vae_checkpoint:
        raise ValueError("Training requires a pretrained VAE checkpoint.")
    
    print(f">>> Loading VAE from: {args.vae_checkpoint}")
    vae_module = VAEModule.load_from_checkpoint(
        checkpoint_path=args.vae_checkpoint, 
        args=args, 
        model=vae_model, 
        map_location=device,
        strict=False
    )

    # Initialize Diffusion architecture
    diffusion_model = GiT(
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

    # Initialize Lightning Module
    diffusion_module = DiffusionModule(args, diffusion_model, vae_module)

    # Check for checkpoint to resume training
    resume_path = None
    if args.diffusion_checkpoint and os.path.exists(args.diffusion_checkpoint):
        print(f">>> Found checkpoint at {args.diffusion_checkpoint}, resuming training...")
        resume_path = args.diffusion_checkpoint

    # Load Single-Cell Data
    print(f">>> Loading data from: {args.data_path}")
    dataset = SCDataset('sc_train', args.data_path) 
    
    # Split Data (8 : 2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_cpus, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_cpus, 
        pin_memory=True
    )

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"diffusion_gene={args.input_size}_latent={args.latent_size}_{timestamp}"
    workdir_path = f"workdir/diffusion/{run_name}"
    os.makedirs(workdir_path, exist_ok=True)
    
    logger = WandbLogger(project='diffusion', save_dir=workdir_path, name=run_name, version='')
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=workdir_path,
        filename="{epoch:03d}-{val_loss:.4f}",
        every_n_epochs=args.save_epoch,
        save_top_k=3,
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, TQDMProgressBar(), lr_monitor]
    
    strategy = DDPStrategy(find_unused_parameters=False) if len(args.devices) > 1 else 'auto'
    
    # Trainer configuration
    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=strategy,
        devices=args.devices,
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.eval_epoch,
        default_root_dir=workdir_path,
        callbacks=callbacks,
    )
    
    # Start Training
    trainer.fit(diffusion_module, train_loader, val_loader, ckpt_path=resume_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    
    parse_data_args(parser)
    parse_model_args(parser)
    parse_training_diffusion_args(parser)
    parse_checkpoint_args(parser)
    parse_transport_args(parser)
    parse_sampling_args(parser)
    
    # Dynamic parsing for sampler specific arguments
    temp_args, _ = parser.parse_known_args()
    if hasattr(temp_args, 'mode'):
        if temp_args.mode == 'ODE':
            parse_ode_args(parser)
        elif temp_args.mode == 'SDE':
            parse_sde_args(parser)

    args = parser.parse_args()
    main(args)