import torch
import pytorch_lightning as pl
from typing import Optional, Tuple

class VAEModule(pl.LightningModule):
    def __init__(self, args, model, freeze_encoder: bool = False):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            self._freeze_encoders()

    def _freeze_encoders(self):
        # Freeze z_encoder
        self.model.z_encoder.requires_grad_(False)
        
        # Freeze l_encoder if exists
        if hasattr(self.model, "l_encoder"):
            self.model.l_encoder.requires_grad_(False)
            
        # Ensure Decoder remains trainable
        self.model.decoder.requires_grad_(True)

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.lr
        )

    def on_train_epoch_start(self):
        # Prevent BN stats drift when encoder is frozen
        if self.freeze_encoder:
            self.model.z_encoder.eval()
            if hasattr(self.model, "l_encoder"):
                self.model.l_encoder.eval()
            self.model.decoder.train()
        else:
            self.model.train()

    # ==============================
    # Training & Validation
    # ==============================
    
    def _get_loss(self, x, y):
        inputs = {
            "X": x,
            "batch": y.long().unsqueeze(1),
            "labels": y.long().unsqueeze(1)
        }
        
        _, _, losses = self.model(inputs)
        return losses.loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._get_loss(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._get_loss(x, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # ==============================
    # Inference Helpers
    # ==============================

    @torch.no_grad()
    def encode_to_latent(self, x, y) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        inputs = {
            "x": x,
            "batch_index": y.long().unsqueeze(1),
        }
        out = self.model.inference(**inputs)
        return out["qz"].mean, out["library"]

    @torch.no_grad()
    def decode_from_latent(self, z, y, l: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.eval()
        
        # Set default library size if missing
        if l is None:
            l = torch.full((len(z), 1), 10000, dtype=torch.float, device=self.device)
        
        inputs = {
            "z": z,
            "library": torch.log(l),
            "batch_index": y.long().unsqueeze(1),
        }
        
        out = self.model.generative(**inputs)            
        return out['px'].mu