import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# model
from scvi.module import VAE
from src.models import GiT

# pl module 
from src.modules import DiffusionModule, VAEModule

from src.data import STDataset
from src.parsing import *

def main(args):
    
    pl.seed_everything(42, workers=True)
    
    # model
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
    
    # pl module
    device = f"cuda:{args.devices[0]}"
    
    if  not args.vae_checkpoint:
        raise ValueError("Please provide a pretrained vae checkpoint.")
    vae_module = VAEModule.load_from_checkpoint(checkpoint_path=args.vae_checkpoint, 
                                                args=args, model=vae_model, map_location=device)
    
    if  not args.diffusion_checkpoint:
        raise ValueError("Please provide a pretrained diffusion checkpoint.")
    diffusion_module = DiffusionModule.load_from_checkpoint(checkpoint_path=args.diffusion_checkpoint, 
                                                            args=args, model=sit_model, vae=vae_module, map_location=device)
    
    # data
    test_dataset = STDataset('st_test', args.input_path, args.cancer_type)

    print(test_dataset.get_num_cell(), test_dataset.get_num_gene())

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_cpus)

    # get mask
    impute_mask = test_dataset.get_impute_mask().to(device)
    
    # impute
    X_imputed, Z_latent = diffusion_module.impute(test_loader, impute_mask, use_sparity_ratio=True, return_latent=True)

    # save
    test_dataset.save_imputed(X_imputed)
    test_dataset.save_latent(Z_latent)
    
    # store
    test_dataset.store_result(args.output_path)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parse_testing_impute_args(parser)
    parse_model_args(parser)
    parse_checkpoint_args(parser)
    parse_transport_args(parser)
    parse_sampling_args(parser)
    parse_data_args(parser)
    

    # decide ODE or SDE
    temp_args, _ = parser.parse_known_args()
    if temp_args.mode == 'ODE':
        parse_ode_args(parser)
    elif temp_args.mode == 'SDE':
        parse_sde_args(parser)

    args = parser.parse_args()
    main(args)