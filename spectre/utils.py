import pydantic
import wandb
import numpy as np
from typing import List, Optional


class WandbRunConfig(pydantic.BaseModel):
    project: str
    run_name: Optional[str] = None
    entity: Optional[str] = None
    group_name: Optional[str] = None
    tags: Optional[List[str]] = None
    log_dir: Optional[str] = None

    # Check that there is a valid wandb entity
    @pydantic.field_validator("entity")
    def valid_entity(cls, entity: Optional[str]) -> str:
        return entity if entity is not None else wandb.Api().default_entity


class WandbCheckpointConfig(WandbRunConfig):

    checkpoint_version: str = "latest"

    @property
    def artifact_name(self) -> str:
        return f"{self.entity}/{self.project}/model-{self.checkpoint_version}"

    
class WandbRunCheckpointConfig(WandbCheckpointConfig):

    run_id: str
    @property
    def full_run_name(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"

    @property
    def artifact_suffix(self) -> str:
        return f"model-{self.run_id}:{self.checkpoint_version}"

    @property
    def artifact_name(self) -> str:
        return f"{self.entity}/{self.project}/{self.artifact_suffix}"


class TrainConfig(pydantic.BaseModel):
    wandb_run_config: Optional[WandbRunConfig] = None
    batch_size: int = 64
    max_epoch_number: int = 10  # Fixed typo here
    warmup_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    seed: int = 42
    gpu_num: int = 0
    log_steps: int = 10
    num_workers: int = 16
    checkpoint_every_n_epochs: int = 5
    profiler: str = "simple"


class HyperParameterScheduler:
    def __init__(self,
                 train_config: TrainConfig,
                 base_value: float,
                 final_value: float,
                 warmup_epochs: int = 0,
                 warmup_initial_value: float = 0):
        """
        Implements a hyperparameter scheduler that can be used to schedule the learning rate, weight decay, etc.

        Parameters
        ----------
        train_config: TrainConfig
            The training configuration object.
        base_value: float
            The initial value of the hyperparameter.
        final_value: float
            The final value of the hyperparameter.
        warmup_epochs: int
            The number of epochs for which the hyperparameter will be linearly increased from the initial value.
        warmup_initial_value: float
            The initial value of the hyperparameter during warmup.
        """
        self.train_config = train_config
        self.scheduling_epochs = train_config.max_epoch_number
        self.base_value = base_value
        self.final_value = final_value
        self.warmup_epochs = warmup_epochs
        self.warmup_initial_value = warmup_initial_value
        
        self.schedule = np.array([])

    def compute_schedule(self, dataloader):
        """
        Computes the schedule for the hyperparameter.

        Parameters
        ----------
        data_loader: DataLoader
            The training data loader.

        Returns
        -------
        None
        """
        iter_per_epoch = len(dataloader)
        warmup_schedule = np.array([])
        warmup_iters = self.warmup_epochs * iter_per_epoch
        if self.warmup_epochs > 0:
            warmup_schedule = np.linspace(self.warmup_initial_value, self.base_value, warmup_iters)
        
        iters = np.arange(self.scheduling_epochs * iter_per_epoch - warmup_iters)
        schedule = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate([warmup_schedule, schedule])
        assert len(schedule) == self.scheduling_epochs * iter_per_epoch
        self.schedule = schedule
        return schedule