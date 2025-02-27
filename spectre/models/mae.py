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
        

class MaskedAutoencoderViT(pl.LightningModule):
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
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.lr_scheduler = lr_scheduler
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed_he = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.patch_embed_codex = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed_he.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.channel_embed = nn.Parameter(torch.zeros(1, n_codex_channels + 1, 1, embed_dim), requires_grad=True)  # learnable channel embedding

        self.blocks = nn.ModuleList([Block(dim=embed_dim,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([Block(dim=decoder_embed_dim,
                                         num_heads=decoder_num_heads,
                                         mlp_ratio=mlp_ratio,
                                         norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_he = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True) # decoder to patch
        self.decoder_pred_codex = nn.Linear(decoder_embed_dim, patch_size**2, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed_he.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.Tensor(pos_embed.astype('float32')).unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed_he.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.Tensor(decoder_pos_embed.astype('float32')).unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_he.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_codex.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.channel_embed, std=.02)

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
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x_he, mask_ratio):
        # embed patches
        x_he = self.patch_embed_he(x_he) # shape [N, L, D]
        
        # add pos embed w/o cls token
        x_he = x_he + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x_he, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # shape: [1, 1, D]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # shape: [N, 1, D]
        x = torch.cat((cls_tokens, x), dim=1) # shape: [N, L+1, D]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # shape: [N, C, L+1, D]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) # shape: [N, L, D_decoder]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # [N, L, D_decoder]
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        N, L, D_decoder = x.shape
        x = x.view(x.shape[0], x.shape[1], x.shape[2])  # [N*C, L, D_decoder]
        x = x + self.decoder_pos_embed
        x = x.view(N, L, D_decoder) # [N, L, D_decoder]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x) # shape: [N, C, L, D_decoder]

        # predictor projection
        x_he = x[:, :, :] # [N, L, D_decoder]
        x_he = self.decoder_pred_he(x_he) # [N, L, p*p*3]

        # remove cls token
        x_he = x_he[:, 1:, :]
        return x_he

    def forward_loss(self, he_imgs, pred_he, mask):
        """
        he_imgs: [N, 3, H, W]
        codex_imgs: [N, C, H, W]
        pred_he: [N, L, p*p*3]
        pred_codex: [N, C, L, p*p]
        mask: [N, C, L], 0 is keep, 1 is remove, 
        """
        target_he = self.patchify_he(he_imgs) # shape: [N, L, p*p*3]
        loss_he = (pred_he - target_he) ** 2  # shape [N, L, p*p*3]
        loss_he = loss_he.mean(dim=-1)  # [N, L], mean loss per patch

        mask_he = mask[:, 0, :] # [N, L]
        loss_he = (loss_he * mask_he).sum() / mask_he.sum()  # mean loss on removed patches
        loss = loss_he
        return loss

    def forward(self, he_imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(he_imgs, mask_ratio)
        pred_he, pred_codex = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(he_imgs, pred_he, mask)
        pred_he_images = self.unpatchify_he(pred_he)
        return loss, pred_he_images
    
    def training_step(self, batch, batch_idx):
        self._update_optimizer()
        self.log('learning_rate', self.lr_scheduler.schedule[self.global_step], prog_bar=False)
        he_imgs = batch
        loss, pred_he_imgs= self(he_imgs, mask_ratio=self.mask_ratio)
        self.log('train_loss', loss)
        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            pred_he = pred_he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            he_to_plot = np.concatenate([image_he, pred_he], axis=1)
            self.logger.experiment.log({"h&e_train": [wandb.Image(he_to_plot)]})
        return loss
    
    def validation_step(self, batch, batch_idx):
        he_imgs = batch
        loss, pred_he_imgs= self(he_imgs, mask_ratio=self.mask_ratio)
        self.log('val_loss', loss)
        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            pred_he = pred_he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            he_to_plot = np.concatenate([image_he, pred_he], axis=1)
            self.logger.experiment.log({"h&e_val": [wandb.Image(he_to_plot)]})
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
    