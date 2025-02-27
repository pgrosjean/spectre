# Pix2Pix Gan with Transformer

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
from spectre.models.spectre_mae import ChannelSpatialAttentionBlock


class SpectrePix2PixGAN(pl.LightningModule):
    def __init__(self,
                 lr_scheduler: HyperParameterScheduler,
                 img_size=224,
                 patch_size=16,
                 n_codex_channels=14,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 disc_depth=8,
                 disc_num_heads=16,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.save_hyperparameters()
        self.lr_scheduler = lr_scheduler
        self.n_codex_channels = n_codex_channels
        # Set the automatic optimization to False
        self.automatic_optimization = False

        # Losses
        self.pix_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.loss_lambda = 100

        # Channel Embedding
        self.channel_embed = nn.Parameter(torch.zeros(1, n_codex_channels + 1, 1, embed_dim), requires_grad=True)  # learnable channel embedding
        torch.nn.init.normal_(self.channel_embed, std=.02)

        # Encoder/Generator
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
                                        for _ in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Discriminator
        self.discriminator_blocks = nn.ModuleList([
            ChannelSpatialAttentionBlock(dim=embed_dim,
                                         num_heads=disc_num_heads,
                                         mlp_ratio=mlp_ratio,
                                         norm_layer=norm_layer)
            for i in range(disc_depth)])
        self.norm = norm_layer(embed_dim)
        self.discriminator_head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

        # Decoder Heads
        self.decoder_pred_channels = nn.ModuleList([nn.Sequential(
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

    def forward_genorator(self, x_he):
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
        out = self.unpatchify_codex(out)
        return out
    
    def forward_discriminator(self, x_he, x_codex):
        # embed patches
        x_he = self.patch_embed_he(x_he) # shape [N, L, D]

        # add pos embed w/o cls token
        x_he = x_he + self.pos_embed[:, 1:, :]

        # x_codex shape: [N, C, H, W]
        N, C, _, _ = x_codex.shape
        x_codex = x_codex.view(x_codex.shape[0]*x_codex.shape[1], x_codex.shape[2], x_codex.shape[3]) # [N*C, H, W]
        x_codex = x_codex.unsqueeze(1) # [N*C, 1, H, W]
        x_codex = self.patch_embed_codex(x_codex) # [N*C, L, D]
        x_codex = x_codex + self.pos_embed[:, 1:, :]
        x_codex = x_codex.view(N, C, x_codex.shape[1], x_codex.shape[2]) # [N, C, L, D]
        
        # add pos embed w/o cls token
        x = torch.concat((x_he.unsqueeze(1), x_codex), dim=1) # [N, C+1, L, D]
        # add the channel embed
        x = x + self.channel_embed

        cls_token = self.cls_token + self.pos_embed[:, :1, :] # shape: [1, 1, D]
        cls_token = cls_token.unsqueeze(0) # shape: [1, 1, 1, D]
        cls_tokens = cls_token.expand(x.shape[0], x.shape[1], -1, -1) # shape: [N, C+1, 1, D]
        x = torch.cat((cls_tokens, x), dim=2) # shape: [N, C+1, L+1, D]
        # apply Transformer blocks
        for blk in self.discriminator_blocks:
            x = blk(x)
        x = self.norm(x) # shape: [N, C+1, L+1, D]
        # Extracting the cls tokens
        x = x[:, 1:, 0, :] # shape: [N, C, D]
        out = self.discriminator_head(x) # shape: [N, C, 1]
        return out
        
    def training_step(self, batch, batch_idx):
        x_he, x_codex = batch
        # Get optimizers manually
        opt_disc, opt_gen = self.optimizers()

        # --- Discriminator update ---
        pred_codex = self.forward_genorator(x_he)
        pred_real = self.forward_discriminator(x_he, x_codex)
        pred_fake = self.forward_discriminator(x_he, pred_codex.detach())
        
        target_real = torch.ones_like(pred_real)
        target_fake = torch.zeros_like(pred_fake)
        
        loss_real = self.bce_loss(pred_real, target_real)
        loss_fake = self.bce_loss(pred_fake, target_fake)
        loss_disc = loss_real + loss_fake
        
        # Update discriminator manually
        opt_disc.zero_grad()
        self.manual_backward(loss_disc, retain_graph=True)
        opt_disc.step()
        
        # --- Generator update ---
        pred_fake = self.forward_discriminator(x_he, pred_codex)
        loss_adv = self.bce_loss(pred_fake, torch.ones_like(pred_fake))
        pix_loss = self.pix_loss(pred_codex, x_codex)
        loss_gen = loss_adv + self.loss_lambda * pix_loss
        
        # Update generator manually
        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        opt_gen.step()
        
        # Logging can be done as usual
        self.log('train_gen_loss', loss_gen, prog_bar=True)
        self.log('train_discrim_loss', loss_disc, prog_bar=True)
        self.log('train_loss', loss_gen + loss_disc, prog_bar=True)

        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = x_he[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = x_codex[0].cpu().detach().numpy()
            pred_codex = pred_codex[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = image_he
            self.logger.experiment.log({"h&e_train": [wandb.Image(he_to_plot)]})
            codex_to_plot = np.concatenate(image_codex, axis=1)
            codex_pred_to_plot = np.concatenate(pred_codex, axis=1)
            images_to_plot = np.concatenate([codex_to_plot, codex_pred_to_plot], axis=0)
            self.logger.experiment.log({"codex_output_train": [wandb.Image(images_to_plot)]})
        
        # Optionally, you can return a dict if you need to log anything further
        return {'loss_disc': loss_disc, 'loss_gen': loss_gen}

    def validation_step(self, batch, batch_idx):
        x_he, x_codex = batch
        pred_codex = self.forward_genorator(x_he)
        pred_real = self.forward_discriminator(x_he, x_codex)
        pred_fake = self.forward_discriminator(x_he, pred_codex)
        pix_loss = self.pix_loss(pred_codex, x_codex)
        
        # Create target labels for real and fake pairs
        target_real = torch.ones_like(pred_real)
        target_fake = torch.zeros_like(pred_fake)

        # Compute binary cross-entropy loss for real and fake predictions
        loss_real = self.bce_loss(pred_real, target_real)
        loss_fake = self.bce_loss(pred_fake, target_fake)

        loss_adv = self.bce_loss(pred_fake, target_real)
        gen_loss = self.loss_lambda * pix_loss + loss_adv

        desc_loss = loss_real + loss_fake

        self.log('val_descrim_loss', desc_loss)
        self.log('val_gen_loss', gen_loss)

        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = x_he[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = x_codex[0].cpu().detach().numpy()
            pred_codex = pred_codex[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = image_he
            self.logger.experiment.log({"h&e_val": [wandb.Image(he_to_plot)]})
            codex_to_plot = np.concatenate(image_codex, axis=1)
            codex_pred_to_plot = np.concatenate(pred_codex, axis=1)
            images_to_plot = np.concatenate([codex_to_plot, codex_pred_to_plot], axis=0)
            self.logger.experiment.log({"codex_output_val": [wandb.Image(images_to_plot)]})
        
    def configure_optimizers(self):
        # Separate parameters by grouping discriminator and generator parameters.
        # Assuming your discriminator parameters are in self.discriminator_blocks and self.discriminator_head
        disc_params = list(self.discriminator_blocks.parameters()) + list(self.discriminator_head.parameters())
        
        # And generator parameters are all the remaining ones.
        gen_params = list(set(self.parameters()) - set(disc_params))
        
        optimizer_disc = torch.optim.AdamW(disc_params, lr=1e-3)
        optimizer_gen = torch.optim.AdamW(gen_params, lr=1e-3)
        
        return [optimizer_disc, optimizer_gen], []

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

    def _update_optimizer(self, opt_index) -> None:
        optimizers = self.optimizers(use_pl_optimizer=True)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr_scheduler.schedule[self.global_step]



#######################################
##### UNET VERSION OF PIX2PIX GAN #####
#######################################

from torchvision import models


class DoubleConv(nn.Module):
    """Applies two consecutive convolutional layers, each followed by ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Contracting path (Encoder)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive path (Decoder)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class SpectrePix2PixGANUNET(pl.LightningModule):
    def __init__(self, lr_scheduler, n_codex_channels=14):
        super().__init__()
        self.lr_scheduler = lr_scheduler
        self.generator = UNet(in_channels=3, out_channels=n_codex_channels)
        self.discriminator = PatchGANDiscriminator(in_channels=n_codex_channels)
        self.pix_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.loss_lambda = 100
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        x_he, x_codex = batch
        # Get optimizers manually
        opt_disc, opt_gen = self.optimizers()

        # --- Discriminator update ---
        pred_codex = self.generator(x_he)
        pred_real = self.discriminator(x_codex)
        pred_fake = self.discriminator(pred_codex)
        
        target_real = torch.ones_like(pred_real)
        target_fake = torch.zeros_like(pred_fake)
        
        loss_real = self.bce_loss(pred_real, target_real)
        loss_fake = self.bce_loss(pred_fake, target_fake)
        loss_disc = loss_real + loss_fake
        
        # Update discriminator manually
        opt_disc.zero_grad()
        self.manual_backward(loss_disc, retain_graph=True)
        opt_disc.step()
        
        # --- Generator update ---
        pred_fake = self.discriminator(pred_codex)
        loss_adv = self.bce_loss(pred_fake, torch.ones_like(pred_fake))
        pix_loss = self.pix_loss(pred_codex, x_codex)
        loss_gen = loss_adv + self.loss_lambda * pix_loss
        
        # Update generator manually
        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        opt_gen.step()
        
        # Logging can be done as usual
        self.log('train_gen_loss', loss_gen, prog_bar=True)
        self.log('train_discrim_loss', loss_disc, prog_bar=True)
        self.log('train_loss', loss_gen + loss_disc, prog_bar=True)

        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = x_he[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = x_codex[0].cpu().detach().numpy()
            pred_codex = pred_codex[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = image_he
            self.logger.experiment.log({"h&e_train": [wandb.Image(he_to_plot)]})
            codex_to_plot = np.concatenate(image_codex, axis=1)
            codex_pred_to_plot = np.concatenate(pred_codex, axis=1)
            images_to_plot = np.concatenate([codex_to_plot, codex_pred_to_plot], axis=0)
            self.logger.experiment.log({"codex_output_train": [wandb.Image(images_to_plot)]})
        
        # Optionally, you can return a dict if you need to log anything further
        return {'loss_disc': loss_disc, 'loss_gen': loss_gen}

    def validation_step(self, batch, batch_idx):
        x_he, x_codex = batch
        pred_codex = self.generator(x_he)
        pred_real = self.discriminator(x_codex)
        pred_fake = self.discriminator(pred_codex)
        pix_loss = self.pix_loss(pred_codex, x_codex)
        
        # Create target labels for real and fake pairs
        target_real = torch.ones_like(pred_real)
        target_fake = torch.zeros_like(pred_fake)

        # Compute binary cross-entropy loss for real and fake predictions
        loss_real = self.bce_loss(pred_real, target_real)
        loss_fake = self.bce_loss(pred_fake, target_fake)

        loss_adv = self.bce_loss(pred_fake, target_real)
        gen_loss = self.loss_lambda * pix_loss + loss_adv

        desc_loss = loss_real + loss_fake

        self.log('val_descrim_loss', desc_loss)
        self.log('val_gen_loss', gen_loss)

        if batch_idx == 0 or batch_idx % 20 == 0:
            image_he = x_he[0].cpu().detach().numpy().transpose(1, 2, 0) # shape: [H, W, 3]
            image_codex = x_codex[0].cpu().detach().numpy()
            pred_codex = pred_codex[0].cpu().detach().numpy()
            maxes_codex = [np.amax([image_codex[i], pred_codex[i]]) for i in range(image_codex.shape[0])]
            image_codex = [image_codex[i] / maxes_codex[i] for i in range(len(image_codex))]
            pred_codex = [pred_codex[i] / maxes_codex[i] for i in range(len(pred_codex))]
            he_to_plot = image_he
            self.logger.experiment.log({"h&e_val": [wandb.Image(he_to_plot)]})
            codex_to_plot = np.concatenate(image_codex, axis=1)
            codex_pred_to_plot = np.concatenate(pred_codex, axis=1)
            images_to_plot = np.concatenate([codex_to_plot, codex_pred_to_plot], axis=0)
            self.logger.experiment.log({"codex_output_val": [wandb.Image(images_to_plot)]})
        
    def configure_optimizers(self):
        # Separate parameters by grouping discriminator and generator parameters.
        # Assuming your discriminator parameters are in self.discriminator_blocks and self.discriminator_head
        disc_params = list(self.discriminator.parameters())
        
        # And generator parameters are all the remaining ones.
        gen_params = list(set(self.parameters()) - set(disc_params))
        
        optimizer_disc = torch.optim.AdamW(disc_params, lr=1e-3)
        optimizer_gen = torch.optim.AdamW(gen_params, lr=1e-3)
        return [optimizer_disc, optimizer_gen], []

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
        optimizers = self.optimizers(use_pl_optimizer=True)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr_scheduler.schedule[self.global_step]
