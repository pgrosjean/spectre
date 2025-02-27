import timm
import torch
from torch import nn
from spectre.utils import HyperParameterScheduler
import pytorch_lightning as pl


class HandEtoMeanProteinAbundance(pl.LightningModule):
    def __init__(self,
                 lr_scheduler: HyperParameterScheduler):
        super(HandEtoMeanProteinAbundance, self).__init__()
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        self.lr_scheduler = lr_scheduler
        self.model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.ff_layer = nn.Sequential(torch.nn.Linear(1536, 128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(128, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(32, 13))
        self.loss = torch.nn.MSELoss()
        self.save_hyperparameters()
    
    def forward(self, x):
        x = self.model(x)
        return self.ff_layer(x)
    
    def training_step(self, batch, batch_idx):
        self._update_optimizer()
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_start(self) -> None:
        """
        Train start hook for scheduler computation.
        """
        # Computing the hyperparameter schedule
        assert self.trainer.max_epochs is not None, "The maximum number of epochs must be specified."
        train_steps = self.trainer.num_training_batches * self.trainer.max_epochs
        self.lr_scheduler.compute_schedule(self.trainer.train_dataloader)
        schedule_length = len(self.lr_scheduler.schedule)
        assert schedule_length != 0 and train_steps <= schedule_length
    
    def _update_optimizer(self) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_scheduler.schedule[self.global_step]
    
    