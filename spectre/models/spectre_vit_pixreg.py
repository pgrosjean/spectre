from functools import partial
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from timm.models.vision_transformer import PatchEmbed
from spectre.models.utils.pos_embed import get_2d_sincos_pos_embed
from spectre.models.utils.mem_eff_block import Block
import wandb
from spectre.utils import HyperParameterScheduler


class SpectreViTPixelRegressor(pl.LightningModule):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 lr_scheduler: HyperParameterScheduler,
                 img_size=224,
                 patch_size=16,
                 n_codex_channels=14,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 individual_decoder=False,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.lr_scheduler = lr_scheduler
        self.patch_embed_he = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.patch_embed_codex = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed_he.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([Block(dim=embed_dim,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=True,
                                         norm_layer=norm_layer)
                                        for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_pred_channels = nn.ModuleList([nn.Linear(embed_dim, patch_size**2 , bias=True) 
                                                    for _ in range(n_codex_channels)]) # decoder to patch
        if individual_decoder:
            print("Using individual decoder heads...")
            self.decoder_pred_channels = nn.ModuleList([nn.Sequential(
                Block(dim=embed_dim,
                       num_heads=8,
                       mlp_ratio=4,
                       qkv_bias=True,
                       norm_layer=norm_layer),
                 nn.Linear(embed_dim, embed_dim, bias=True),
                 nn.ReLU(),
                 nn.Linear(embed_dim, embed_dim, bias=True),
                 nn.ReLU(),
                 nn.Linear(embed_dim, patch_size**2, bias=True))
                for _ in range(n_codex_channels)])
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed_he.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.Tensor(pos_embed.astype('float32')).unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_he.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_codex.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify_he(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed_he.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x) # shape [N, H, W, p, q, 3]
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3)) # shape [N, L, p*p*3]
        return x
    
    def patchify_codex(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, C, L, patch_size**2)

        Perfroming channel specific patchification
        """
        p = self.patch_embed_codex.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nchwpq', x) # shape: [N, C, H, W, p, q]
        x = x.reshape(shape=(imgs.shape[0], imgs.shape[1], h * w, p**2)) # shape: [N, C, L, p*p]
        return x

    def unpatchify_he(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed_he.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p)) # shape [N, 3, H, W]
        return imgs
    
    def unpatchify_codex(self, x):
        """
        x: (N, C, L, patch_size**2)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed_codex.patch_size[0]
        h = w = int(x.shape[2]**.5)
        assert h * w == x.shape[2]
        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p)) # shape: [N, C, H, W, p, q]
        x = torch.einsum('nchwpq->nchpwq', x) # shape: [N, C, H, W, p, q]
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], h * p, h * p)) # shape: [N, C, H, W]
        return imgs

    def forward_encoder(self, x_he):
        # embed patches
        x_he = self.patch_embed_he(x_he) # shape [N, L, D]
        # add pos embed w/o cls token
        x_he = x_he + self.pos_embed[:, 1:, :] # shape: 
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # shape: [1, 1, D]
        cls_tokens = cls_token.expand(x_he.shape[0], -1, -1) # shape: [N, 1, D]
        x = torch.cat((cls_tokens, x_he), dim=1) # shape: [N, L+1, D]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # shape: [N, L+1, D]
        out = torch.stack([decoder_proj(x) for decoder_proj in self.decoder_pred_channels], dim=0) # shape: [C, N, L+1, p*p]
        out = out.permute(1, 0, 2, 3) # shape: [N, C, L+1, p*p]
        # removing CLS token
        out = out[:, :, 1:, :] # shape: [N, C, L, p*p]
        return out

    def forward_loss(self, codex_imgs, pred_codex):
        target_codex = self.patchify_codex(codex_imgs) # shape: [N, C, L, p*p]
        loss_codex = (pred_codex - target_codex) ** 2  # shape [N, C, L, p*p]
        loss = loss_codex.mean()
        return loss

    def forward(self, he_imgs, codex_imgs):
        pred_codex = self.forward_encoder(he_imgs)
        loss = self.forward_loss(codex_imgs, pred_codex)
        pred_codex_images = self.unpatchify_codex(pred_codex)
        return loss, pred_codex_images
    
    def training_step(self, batch, batch_idx):
        self._update_optimizer()
        self.log('learning_rate', self.lr_scheduler.schedule[self.global_step], prog_bar=False)
        he_imgs, codex_imgs = batch
        loss, pred_codex_imgs = self(he_imgs, codex_imgs)
        self.log('train_loss', loss)
        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = codex_imgs[0].cpu().detach().numpy()
            pred_codex = pred_codex_imgs[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = image_he
            self.logger.experiment.log({"h&e_train": [wandb.Image(he_to_plot)]})
            codex_to_plot = np.concatenate(image_codex, axis=1)
            codex_pred_to_plot = np.concatenate(pred_codex, axis=1)
            images_to_plot = np.concatenate([codex_to_plot, codex_pred_to_plot], axis=0)
            self.logger.experiment.log({"codex_output_train": [wandb.Image(images_to_plot)]})
        return loss
    
    def validation_step(self, batch, batch_idx):
        he_imgs, codex_imgs = batch
        loss, pred_codex_imgs = self(he_imgs, codex_imgs)
        self.log('val_loss', loss)
        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = codex_imgs[0].cpu().detach().numpy()
            pred_codex = pred_codex_imgs[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = image_he
            self.logger.experiment.log({"h&e_val": [wandb.Image(he_to_plot)]})
            codex_to_plot = np.concatenate(image_codex, axis=1)
            codex_pred_to_plot = np.concatenate(pred_codex, axis=1)
            images_to_plot = np.concatenate([codex_to_plot, codex_pred_to_plot], axis=0)
            self.logger.experiment.log({"codex_output_val": [wandb.Image(images_to_plot)]})
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
    