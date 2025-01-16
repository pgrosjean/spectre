import zarr
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Union
from spectre.utils import TrainConfig
import lightning as L
from torchvision.transforms import Normalize

ZARR_FILE = "/scratch/pgrosjean/registered_images_matched_channels.zarr"

# Data Utilities
def generate_dataloader(train_config: TrainConfig,
                        dataset: Dataset,
                        shuffle: bool = True) -> DataLoader:
    """
    This function generates a PyTorch DataLoader object for training a model.

    Parameters
    ----------
    train_config : TrainConfig
        An object containing the training configuration parameters.
    dataset : Dataset
        The dataset object to be used for training.
    
    Returns
    -------
    DataLoader
        A PyTorch DataLoader object for training the model.
    """
    num_workers = train_config.num_workers
    batch_size = train_config.batch_size
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader


# Datasets
class HandEtoCODEXDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, codex_keys: Union[List[str],None]=None):
        self.zarr_root = zarr.open(ZARR_FILE, mode='r')
        self.transform = transform
        self.codex_keys = list(self.zarr_root.keys())
        self.codex_keys = [key for key in self.codex_keys if key != 'aligned_codex_7']
        if codex_keys is not None:
            assert set(codex_keys).issubset(set(self.codex_keys)), "Invalid codex_keys"
            self.codex_keys = [key for key in self.codex_keys if key in codex_keys]
        self.codex_indices = self._get_codex_indices()

    def _get_codex_indices(self):
        codex_indices = []
        for key in self.codex_keys:
            n_samples = self.zarr_root[key]['codex'].shape[0]
            codex_indices.extend([(key, idx) for idx in range(n_samples)])
        return codex_indices

    def __len__(self):
        return len(self.codex_indices)

    def __getitem__(self, idx):
        key, idx = self.codex_indices[idx]
        codex = np.array(self.zarr_root[key]['codex'][idx]).astype('float32')
        he = np.array(self.zarr_root[key]['he'][idx]).astype('float32')
        if self.transform is not None:
            he = torch.Tensor(he)
            he = self.transform(he)
        return he, torch.Tensor(codex)
    

class HandEtoProteinExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, codex_keys: Union[List[str],None]=None):
        self.zarr_root = zarr.open(ZARR_FILE, mode='r')
        self.transform = transform
        self.codex_keys = list(self.zarr_root.keys())
        self.codex_keys = [key for key in self.codex_keys if key != 'aligned_codex_7']
        if codex_keys is not None:
            assert set(codex_keys).issubset(set(self.codex_keys)), "Invalid codex_keys"
            self.codex_keys = [key for key in self.codex_keys if key in codex_keys]
        self.codex_indices = self._get_codex_indices()

    def _get_codex_indices(self):
        codex_indices = []
        for key in self.codex_keys:
            n_samples = self.zarr_root[key]['codex'].shape[0]
            codex_indices.extend([(key, idx) for idx in range(n_samples)])
        return codex_indices
    
    def _normalize_codex(self, codex):
        # codex shape: [13, 244, 244]
        max_vals = torch.tensor([349.2795,
                     325.7382,
                     904.5319,
                     252.3597,
                     190.8374,
                     93.6197,
                     171.4191, 
                     135.3005, 
                     32.8869,
                     131.8111,
                     180.8844,
                     176.6160,
                     344.6268])
        codex = codex.permute(1, 2, 0)
        codex = codex / max_vals
        codex = codex.permute(2, 0, 1)
        return codex

    def __len__(self):
        return len(self.codex_indices)

    def __getitem__(self, idx):
        key, idx = self.codex_indices[idx]
        codex = np.array(self.zarr_root[key]['codex'][idx]).astype('float32')
        codex = self._normalize_codex(torch.Tensor(codex))
        he = np.array(self.zarr_root[key]['he'][idx]).astype('float32')
        he = he / 255.0
        if self.transform is not None:
            he = torch.Tensor(he)
            he = self.transform(he)
        return he, codex.view(codex.shape[0], -1).mean(dim=-1)
    

# DataModules
class HEProtDataModule(L.LightningDataModule):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.train_config = train_config
        self.train_ds = HandEtoProteinExpressionDataset(
                codex_keys=['aligned_codex_1', 'aligned_codex_2', 'aligned_codex_3'],
                transform=Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            )
        self.val_ds = HandEtoProteinExpressionDataset(
                codex_keys=['aligned_codex_5'],
                transform=Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            )

    def train_dataloader(self):
        """
        Returns the training DataLoader after ensuring `setup` has been called.
        """
        if self.train_ds is None:
            raise ValueError("Training dataset has not been set up. Did you forget to call `setup('fit')`?")
        return generate_dataloader(self.train_config, self.train_ds, shuffle=True)

    def val_dataloader(self):
        """
        Returns the validation DataLoader after ensuring `setup` has been called.
        """
        if self.val_ds is None:
            raise ValueError("Validation dataset has not been set up. Did you forget to call `setup('validate')`?")
        return generate_dataloader(self.train_config, self.val_ds, shuffle=False)


class HEtoCODEXDataModule(L.LightningDataModule):
    def __init__(self,
                 train_config: TrainConfig):
        super().__init__()
        self.train_config = train_config

    def train_dataloader(self):
        train_ds = HandEtoCODEXDataset(codex_keys=['aligned_codex_1', 'aligned_codex_2', 'aligned_codex_3'],
                                       transform=Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        return generate_dataloader(self.train_config, train_ds, shuffle=True)
    
    def val_dataloader(self):
        val_ds = HandEtoCODEXDataset(codex_keys=['aligned_codex_5'],
                                     transform=Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        return generate_dataloader(self.train_config, val_ds, shuffle=False)