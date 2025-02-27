import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import numpy as np
import torch.nn.functional as F



class FlexibleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super(FlexibleUNet, self).__init__()
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

        # Per-channel independent processing
        self.channel_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=1),  # Single-channel output for each processor
                nn.Conv2d(16, 1, kernel_size=1)
            )
            for _ in range(out_channels)
        ])

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

        # Independent per-channel processing
        outputs = []
        for i, processor in enumerate(self.channel_processors):
            # Extract the i-th channel and apply the processor
            channel_output = processor(dec1)
            outputs.append(channel_output)

        # Combine outputs along the channel dimension
        return torch.cat(outputs, dim=1)


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

class HandEtoCODEX(pl.LightningModule):
    def __init__(self, lr_scheduler, in_channels=3, out_channels=13):
        super(HandEtoCODEX, self).__init__()
        self.lr_scheduler = lr_scheduler
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = FlexibleUNet(in_channels=self.in_channels, out_channels=self.out_channels)
        # self.model = UNet(in_channels=self.in_channels, out_channels=self.out_channels)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self._update_optimizer()
        self.log('learning_rate', self.lr_scheduler.schedule[self.global_step], prog_bar=False)
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # y_hat_channels = [y_hat[:, i, :, :].detach() for i in range(y_hat.shape[1])]
        # y_channels = [y[:, i, :, :].detach() for i in range(y.shape[1])]
        # channel_specifc_losses = [F.mse_loss(y_hat_channels[i], y_channels[i]) for i in range(len(y_hat_channels))]
        # for i, loss_val in enumerate(channel_specifc_losses):
        #     self.log(f'train_loss_channel_{i}', loss_val, prog_bar=False)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        # plotting the input and output prediction for the first batch
        if batch_idx == 0:
            image_in = x[0].cpu().detach().numpy().transpose(1, 2, 0)
            images_out = [y_hat[0].cpu().detach().numpy().transpose(1, 2, 0)[:, :, i] for i in range(y_hat.shape[1])]
            images_gt = [y[0].cpu().detach().numpy().transpose(1, 2, 0)[:, :, i] for i in range(y.shape[1])]
            maxes = [np.max([images_out[i], images_gt[i]]) for i in range(len(images_out))]
            mins = [np.min([images_out[i], images_gt[i]]) for i in range(len(images_out))]
            images_out = [(images_out[i] - mins[i]) / (maxes[i] - mins[i]) for i in range(len(images_out))]
            images_gt = [(images_gt[i] - mins[i]) / (maxes[i] - mins[i]) for i in range(len(images_gt))]
            self.logger.experiment.log({
                "input_train": [wandb.Image(image_in)]})
            for i in range(len(images_out)):
                # concatenating along X axis all the images_out
                images_to_plot_out = np.concatenate(images_out, axis=1)
                images_to_plot_gt = np.concatenate(images_gt, axis=1)
                # concatenating along Y axis the images_out and images_gt
                images_to_plot = np.concatenate([images_to_plot_out, images_to_plot_gt], axis=0)
                self.logger.experiment.log({"predicted_and_ground_truth_train": [wandb.Image(images_to_plot)]})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        # y_hat_channels = [y_hat[:, i, :, :].detach() for i in range(y_hat.shape[1])]
        # y_channels = [y[:, i, :, :].detach() for i in range(y.shape[1])]
        # channel_specifc_losses = [F.mse_loss(y_hat_channels[i], y_channels[i]).detach() for i in range(len(y_hat_channels))]
        # for i, loss_val in enumerate(channel_specifc_losses):
        #     self.log(f'val_loss_channel_{i}', loss_val, prog_bar=False)
        if batch_idx == 3:
            image_in = x[-1].cpu().detach().numpy().transpose(1, 2, 0)
            images_out = [y_hat[-1].cpu().detach().numpy().transpose(1, 2, 0)[:, :, i] for i in range(y_hat.shape[1])]
            images_gt = [y[-1].cpu().detach().numpy().transpose(1, 2, 0)[:, :, i] for i in range(y.shape[1])]
            maxes = [np.max([images_out[i], images_gt[i]]) for i in range(len(images_out))]
            images_out = [images_out[i] / maxes[i] for i in range(len(images_out))]
            images_gt = [images_gt[i] / maxes[i] for i in range(len(images_gt))]
            self.logger.experiment.log({
                "input_val": [wandb.Image(image_in)]})
            for i in range(len(images_out)):
                # concatenating along X axis all the images_out
                images_to_plot_out = np.concatenate(images_out, axis=1)
                images_to_plot_gt = np.concatenate(images_gt, axis=1)
                # concatenating along Y axis the images_out and images_gt
                images_to_plot = np.concatenate([images_to_plot_out, images_to_plot_gt], axis=0)
                self.logger.experiment.log({"predicted_and_ground_truth_val": [wandb.Image(images_to_plot)]})
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