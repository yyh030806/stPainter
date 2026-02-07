import os
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from scvi.module import VAE 
from src.modules import VAEModule 
from src.data import SCDataset
from src.parsing import (
    parse_training_vae_args,
    parse_model_args,
    parse_data_args
)

def main(args):
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    print(f">>> Mode: Training VAE")
    print(f">>> Data Path: {args.data_path}")

    # Initialize Base VAE Model
    model = VAE(
        n_input=args.input_size,
        n_batch=args.num_classes,
        n_hidden=args.hidden_size_vae,
        n_latent=args.latent_size,
        n_layers=args.num_layer,
    )
    
    # Initialize Lightning Module Wrapper
    vae_module = VAEModule(args, model)

    # Load Dataset
    dataset = SCDataset('sc_train', args.data_path)
    
    # Split Dataset(8 : 2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create DataLoaders
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

    # Logging and Checkpoint Configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vae_gene={args.input_size}_latent={args.latent_size}_{timestamp}"
    workdir_path = f"workdir/vae/{run_name}"
    os.makedirs(workdir_path, exist_ok=True)
    
    logger = WandbLogger(
        project='vae', 
        save_dir=workdir_path, 
        name=run_name, 
        version=''
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=workdir_path,
        filename="vae-{epoch:03d}-{val_loss:.4f}",
        every_n_epochs=args.save_epoch,
        save_top_k=3,
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, TQDMProgressBar(), lr_monitor]
    
    # Configure Strategy
    strategy = DDPStrategy(find_unused_parameters=False) if len(args.devices) > 1 else 'auto'
    
    # Initialize Trainer
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
    trainer.fit(vae_module, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parse_data_args(parser)
    parse_model_args(parser)
    parse_training_vae_args(parser)

    args = parser.parse_args()
    main(args)