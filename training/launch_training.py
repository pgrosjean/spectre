import os
from pathlib import Path

import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from lightning.pytorch.loggers import WandbLogger
import hydra
import tempfile
import wandb
import lightning as L


# Setting up the environment for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["WANDB_DIR"] = "/scratch/pgrosjean/wandb/"
os.environ["WANDB_ARTIFACT_DIR"] = "/scratch/pgrosjean/wandb/artifacts/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/pgrosjean/wandb/cache/"
os.environ["WANDB_DATA_DIR"] = "/scratch/pgrosjean/wandb/data/"


def main():
    # Seeding everything to ensure reproducibility
    pl.seed_everything(1)
    
    # Argparsing
    desc = "Script for Training a regressor model on protein sequence"
    parser = ArgumentParser(
        description=desc, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="config file name without .yaml extension")
    parser.add_argument("--log_dir", default="/scratch/pgrosjean/wandb/" , type=Path, help="Directory to save logs")
    args = parser.parse_args()
    config_path = Path(__file__).parent / "config"
    
    full_path = Path.cwd() / config_path
    relative_path = os.path.relpath(full_path, Path(__file__).parent)

    with hydra.initialize(version_base=None, config_path=str(relative_path)):
        config = hydra.compose(config_name=str(args.config))

    model_config = config.model_config
    train_config = config.train_config
    
    wandb_config = config.wandb_config
    data_config = config.data_config

    # Instantiating the train config with defaults
    train_config = hydra.utils.instantiate(train_config)

    # Generating DataLoaders
    print("Generating Training and Validation DataSets...")
    data_module = hydra.utils.instantiate(data_config)
    
    # Instantiating Model
    print("Instantiating the model...")
    model = hydra.utils.instantiate(model_config)

    # Initializing Wandb Logger
    wandb_config = hydra.utils.instantiate(wandb_config)
    
    log_dir = tempfile.mkdtemp(dir=args.log_dir)
    logger = WandbLogger(project=wandb_config.project,
                         name=wandb_config.run_name,
                         log_model=True,
                         group=wandb_config.group_name,
                         entity=wandb_config.entity,
                         tags=wandb_config.tags,
                         dir=log_dir,
                         save_dir=log_dir)

    # Creating Trainer from argparse args
    if train_config.gpu_num == -1:
        gpu_num = -1
    else:
        gpu_num = [train_config.gpu_num]

    callbacks = [ModelCheckpoint(every_n_epochs=train_config.checkpoint_every_n_epochs,
                                dirpath='/scratch/pgrosjean/wandb/checkpoints/',
                                save_top_k=2,
                                monitor="train_loss",
                                mode="min")
                ]

    assert torch.cuda.is_available(), "CUDA is not available and is required for training"

    trainer = L.Trainer(accelerator="gpu",
                         devices=gpu_num,
                         logger=logger,
                         enable_checkpointing=True,
                         profiler="simple",
                         callbacks=callbacks,
                         max_epochs=train_config.max_epoch_number,
                         log_every_n_steps=train_config.log_steps,
                         precision=32)
    # Training the model
    print("Training Model...")
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()