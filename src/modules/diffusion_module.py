import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
from typing import Optional, Dict, Any

from ..transport import create_transport, Sampler

class DiffusionModule(pl.LightningModule):
    # t=0: noise, t=1: data
    def __init__(self, args, model, vae):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Models
        self.model = model
        self.vae = vae
        self._freeze(self.vae)
        
        # EMA setup
        self.ema_decay = getattr(self.hparams, 'ema_decay', 0.9999)
        self.ema_model = deepcopy(self.model)
        self._freeze(self.ema_model)

        # Transport & Sampler
        self.transport = create_transport(
            path_type=self.hparams.path_type,
            prediction=self.hparams.prediction,
            loss_weight=getattr(self.hparams, 'loss_weight', None),
            train_eps=getattr(self.hparams, 'train_eps', None),
            sample_eps=getattr(self.hparams, 'sample_eps', None),
        )
        self.sampler = Sampler(self.transport)

        # Preload Sparsity Data (IO Optimization)
        self.sparsity_ratio = None
        path = getattr(self.hparams, 'gene_sparsity_ratio_file_path', None)
        if path:
            try:
                self.sparsity_ratio = pd.read_csv(path, index_col=0).squeeze()
            except Exception as e:
                print(f"Warning: Failed to load sparsity file: {e}")

    def _freeze(self, module):
        module.requires_grad_(False)
        module.eval()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0)

    # ==============================
    # Training Loop
    # ==============================
    def training_step(self, batch, batch_idx):
        x, y = batch
        z, _ = self.vae.encode_to_latent(x, y)
        
        loss_dict = self.transport.training_losses(self.model, z, dict(y=y))
        loss = loss_dict["loss"].mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z, _ = self.vae.encode_to_latent(x, y)
        
        loss_dict = self.transport.training_losses(self.model, z, dict(y=y))
        loss = loss_dict["loss"].mean() 
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_ema()

    @torch.no_grad()
    def _update_ema(self):
        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())
        for name, param in model_params.items():
            if name in ema_params:
                ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    # ==============================
    # Sampling & Helpers
    # ==============================
    def get_sample_fn(self, mode, cfg_scale, t_forward=None):
        kwargs = {'t_forward': t_forward}
        
        # Load args based on mode
        keys = []
        if mode == "ODE":
            keys = ['sampling_method', 'num_steps', 'atol', 'rtol', 'reverse']
        elif mode == "SDE":
            keys = ['sampling_method', 'diffusion_form', 'diffusion_norm', 'last_step', 'last_step_size', 'num_steps']
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for k in keys:
            if hasattr(self.hparams, k): kwargs[k] = getattr(self.hparams, k)

        # Return function
        if mode == "ODE":
            if kwargs.get('likelihood', False) and cfg_scale != 1.0:
                raise ValueError("Likelihood sampling requires cfg_scale=1.0")
            return self.sampler.sample_ode_likelihood(**kwargs) if kwargs.get('likelihood') else self.sampler.sample_ode(**kwargs)
        return self.sampler.sample_sde(**kwargs)

    @torch.no_grad()
    def sample(self, z, y, mode=None, cfg_scale=None):
        self.ema_model.eval()
        mode = mode or self.hparams.mode
        cfg_scale = cfg_scale or self.hparams.cfg_scale
        
        sample_fn = self.get_sample_fn(mode, cfg_scale)
        
        # CFG Setup
        if cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            y = torch.cat([y, torch.zeros_like(y)], dim=0) # Assume 0 is null
            model_fn = self.ema_model.forward_with_cfg
            kwargs = dict(y=y, cfg_scale=cfg_scale)
        else:
            model_fn = self.ema_model.forward
            kwargs = dict(y=y)

        samples = sample_fn(z, model_fn, **kwargs)[-1]
        return samples.chunk(2, dim=0)[0] if cfg_scale > 1.0 else samples

    # ==============================
    # Imputation
    # ==============================
    def _apply_sparsity_constraint(self, data, cancer_type):
        """Sets bottom S% values to 0 per gene."""
        if self.sparsity_ratio is None: return data
        
        # Get ratios
        vals = None
        if isinstance(self.sparsity_ratio, pd.DataFrame) and cancer_type in self.sparsity_ratio:
            vals = self.sparsity_ratio[cancer_type].values
        elif isinstance(self.sparsity_ratio, pd.Series):
            vals = self.sparsity_ratio.values
            
        if vals is None or len(vals) != data.shape[1]: return data

        # Apply per gene
        for i, ratio in enumerate(vals):
            if ratio <= 0: continue
            if ratio >= 1.0:
                data[:, i] = 0
            else:
                thresh = np.percentile(data[:, i], ratio * 100)
                data[data[:, i] < thresh, i] = 0
        return data

    @torch.no_grad()
    def impute(self, loader, known_mask, use_sparity_ratio=True, return_latent=True):
        res_x, res_z = [], []
        if isinstance(known_mask, torch.Tensor): known_mask = known_mask.to(self.device)

        for x, y in tqdm(loader, desc="Imputing"):
            x, y = x.to(self.device), y.to(self.device)
            x_imp, z_lat = self.impute_batch(x, known_mask, y, t_forward=self.hparams.t_forward)
            res_x.append(x_imp.cpu().numpy())
            res_z.append(z_lat.cpu().numpy())

        X_out = np.concatenate(res_x, axis=0)
        Z_out = np.concatenate(res_z, axis=0)
        
        if use_sparity_ratio:
            X_out = self._apply_sparsity_constraint(X_out, self.hparams.cancer_type)
            
        return (X_out, Z_out) if return_latent else X_out

    @torch.no_grad()
    def impute_batch(self, x, x_mask, y, t_forward=0.9, mode=None, cfg_scale=None):
        self.ema_model.eval()
        mode = mode or self.hparams.mode
        cfg_scale = cfg_scale or self.hparams.cfg_scale
        
        # Encode masked input
        z_guide, _ = self.vae.encode_to_latent(x * x_mask, y)
        
        # Forward Diffuse
        t_ts = torch.full((len(x),), t_forward, device=self.device)
        _, z_t, _ = self.transport.path_sampler.plan(t_ts, torch.randn_like(z_guide), z_guide)
        
        # Denoise
        sample_fn = self.get_sample_fn(mode, cfg_scale, t_forward)
        
        if cfg_scale > 1.0:
            z_in = torch.cat([z_t, z_t], 0)
            y_in = torch.cat([y, torch.zeros_like(y)], dim=0)
            fn = self.ema_model.forward_with_cfg
            kwargs = dict(y=y_in, cfg_scale=cfg_scale)
        else:
            z_in = z_t
            fn = self.ema_model.forward
            kwargs = dict(y=y)
            
        z_1 = sample_fn(z_in, fn, **kwargs)[-1]
        if cfg_scale > 1.0: z_1, _ = z_1.chunk(2, dim=0)
        
        # Decode
        x_rec = self.vae.decode_from_latent(z_1, y)
        return x_rec, z_1