from functools import partial
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from timm.models.vision_transformer import PatchEmbed
from spectre.models.utils.pos_embed import get_2d_sincos_pos_embed
from spectre.models.utils.mem_eff_block import Block
import wandb
from spectre.utils import HyperParameterScheduler

class ChannelSpatialAttentionBlock(nn.Module):
    """
    """
    def __init__(self,
                 dim=1024,
                 num_heads=16,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_transformer = Block(dim=dim,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=True,
                                         norm_layer=norm_layer)
        self.spatial_transformer = Block(dim=dim,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=True,
                                        norm_layer=norm_layer)
        
    def forward(self, x):
        # Expected input shape: [batch, n_channels, n_patches, embed_dim]
        x = x.permute(0, 2, 1, 3)  # [batch, n_patches, n_channels, embed_dim]
        B, P, C, E = x.shape
        x = x.reshape(shape=(B * P, C, E))
        x = self.channel_transformer(x)
        x = x.reshape(shape=(B, P, C, E))
        x = x.permute(0, 2, 1, 3) # [batch, n_channels, n_patches, embed_dim]
        x = x.reshape(shape=(B * C, P, E))
        x = self.spatial_transformer(x)
        x = x.reshape(shape=(B, C, P, E))
        return x
        

class SpectreMaskedAutoencoderViT(L.LightningModule):
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

        self.blocks = nn.ModuleList([
            ChannelSpatialAttentionBlock(dim=embed_dim,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.channel_embed_decoder = nn.Parameter(torch.zeros(1, n_codex_channels + 1, 1, decoder_embed_dim), requires_grad=True)

        self.decoder_blocks = nn.ModuleList([
            ChannelSpatialAttentionBlock(dim=decoder_embed_dim,
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
    
    def channel_wise_random_masking(self, x, mask_ratio):
        """
        Perform channel-wise masking by setting the channel to zero.
        x: [N, C, L, D], sequence
        """
        N, C, L, D = x.shape
        noise = torch.rand(N, C, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample per channel
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :int(L * (1 - mask_ratio))]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, C, L], device=x.device)
        mask[:, :, :int(L * (1 - mask_ratio))] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)

        # x_masked_shape: [N, C, L, D], mask_shape: [N, C, L], ids_restore_shape: [N, C, L]
        return x_masked, mask, ids_restore

    def forward_encoder(self, x_he, x_codex, mask_ratio):
        # embed patches
        x_he = self.patch_embed_he(x_he) # shape [N, L, D]

        # x_codex shape: [N, C, H, W]
        N, C, _, _ = x_codex.shape
        x_codex = x_codex.view(x_codex.shape[0]*x_codex.shape[1], x_codex.shape[2], x_codex.shape[3]) # [N*C, H, W]
        x_codex = x_codex.unsqueeze(1) # [N*C, 1, H, W]
        x_codex = self.patch_embed_codex(x_codex) # [N*C, L, D]
        
        # add pos embed w/o cls token
        x_he = x_he + self.pos_embed[:, 1:, :]
        x_codex = x_codex + self.pos_embed[:, 1:, :]
        x_codex = x_codex.view(N, C, x_codex.shape[1], x_codex.shape[2]) # [N, C, L, D]
        x = torch.concat((x_he.unsqueeze(1), x_codex), dim=1) # [N, C+1, L, D]
        # add the channel embed
        x = x + self.channel_embed 

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.channel_wise_random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # shape: [1, 1, D]
        cls_token = cls_token.unsqueeze(0) # shape: [1, 1, 1, D]
        cls_tokens = cls_token.expand(x.shape[0], x.shape[1], -1, -1) # shape: [N, C+1, 1, D]
        x = torch.cat((cls_tokens, x), dim=2) # shape: [N, C+1, L+1, D]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # shape: [N, C, L+1, D]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) # shape: [N, C, L, D_decoder]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], ids_restore.shape[2] + 1 - x.shape[2], 1)
        x_ = torch.cat([x[:, :, 1:, :], mask_tokens], dim=2)  # no cls token
        x_ = torch.gather(x_, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, x.shape[3]))  # unshuffle
        x = torch.cat([x[:, :, :1, :], x_], dim=2)  # append cls token

        # add pos embed
        N, C, L, D_decoder = x.shape
        x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])  # [N*C, L, D_decoder]
        x = x + self.decoder_pos_embed
        x = x.view(N, C, L, D_decoder) # [N, C, L, D_decoder]

        # add channel embed
        x = x + self.channel_embed_decoder

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x) # shape: [N, C, L, D_decoder]

        # predictor projection
        x_he = x[:, 0, :, :] # [N, L, D_decoder]
        x_he = self.decoder_pred_he(x_he) # [N, L, p*p*3]
        x_codex = x[:, 1:, :, :]
        x_codex = self.decoder_pred_codex(x_codex) # [N, C, L, p*p]

        # remove cls token
        assert len(x_he.shape) == 3 and len(x_codex.shape) == 4, f"{x_he.shape}, {x_codex.shape}"
        x_he = x_he[:, 1:, :]
        x_codex = x_codex[:, :, 1:, :]
        return x_he, x_codex

    def forward_loss(self, he_imgs, codex_imgs, pred_he, pred_codex, mask):
        """
        he_imgs: [N, 3, H, W]
        codex_imgs: [N, C, H, W]
        pred_he: [N, L, p*p*3]
        pred_codex: [N, C, L, p*p]
        mask: [N, C, L], 0 is keep, 1 is remove, 
        """
        target_he = self.patchify_he(he_imgs) # shape: [N, L, p*p*3]
        target_codex = self.patchify_codex(codex_imgs) # shape: [N, C, L, p*p]
        loss_he = (pred_he - target_he) ** 2  # shape [N, L, p*p*3]
        loss_he = loss_he.mean(dim=-1)  # [N, L], mean loss per patch
        loss_codex = (pred_codex - target_codex) ** 2  # shape [N, C, L, p*p]
        loss_codex = loss_codex.mean(dim=-1) # [N, C, L], mean loss per patch

        mask_he = mask[:, 0, :] # [N, L]
        mask_codex = mask[:, 1:, :] # [N, C, L]
        loss_he = (loss_he * mask_he).sum() / mask_he.sum()  # mean loss on removed patches
        loss_codex = (loss_codex * mask_codex).sum() / mask_codex.sum()  # mean loss on removed patches
        loss = loss_he + loss_codex
        return loss

    def forward(self, he_imgs, codex_imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(he_imgs, codex_imgs, mask_ratio)
        pred_he, pred_codex = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(he_imgs, codex_imgs, pred_he, pred_codex, mask)
        pred_he_images = self.unpatchify_he(pred_he)
        pred_codex_images = self.unpatchify_codex(pred_codex)
        return loss, pred_he_images, pred_codex_images
    
    def training_step(self, batch, batch_idx):
        self._update_optimizer()
        self.log('learning_rate', self.lr_scheduler.schedule[self.global_step], prog_bar=False)
        he_imgs, codex_imgs = batch
        loss, pred_he_imgs, pred_codex_imgs = self(he_imgs, codex_imgs, mask_ratio=self.mask_ratio)
        self.log('train_loss', loss)
        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            pred_he = pred_he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = codex_imgs[0].cpu().detach().numpy()
            pred_codex = pred_codex_imgs[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = np.concatenate([image_he, pred_he], axis=1)
            self.logger.experiment.log({"h&e_train": [wandb.Image(he_to_plot)]})
            codex_to_plot = np.concatenate(image_codex, axis=1)
            codex_pred_to_plot = np.concatenate(pred_codex, axis=1)
            images_to_plot = np.concatenate([codex_to_plot, codex_pred_to_plot], axis=0)
            self.logger.experiment.log({"codex_output_train": [wandb.Image(images_to_plot)]})
        return loss
    
    def validation_step(self, batch, batch_idx):
        he_imgs, codex_imgs = batch
        loss, pred_he_imgs, pred_codex_imgs = self(he_imgs, codex_imgs, mask_ratio=self.mask_ratio)
        self.log('val_loss', loss)
        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            pred_he = pred_he_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = codex_imgs[0].cpu().detach().numpy()
            pred_codex = pred_codex_imgs[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = np.concatenate([image_he, pred_he], axis=1)
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
    