"""
TEMPLATE DATA UTILITIES
=======================
Helper functions for instantiating datasets, dataloaders, and
moving batch transforms to the correct device.
"""

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch transforms to the target device.

    Batch transforms are applied on the batch which may already be on GPU,
    so the transforms themselves must also be on that device.

    Args:
        batch_transforms (dict[str, dict[str, Callable]]): nested dict of
            transforms, keyed by batch field name then transform name.
        device (str): target device string (e.g., "cuda", "cpu", "auto").

    Returns:
        dict[str, dict[str, Callable]]: transforms moved to device.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)
    return batch_transforms


def get_datasets(config):
    """
    Create dataset instances for each partition defined in config.

    Reads config.datasets.{partition} and instantiates each via Hydra.

    Args:
        config (DictConfig): Hydra experiment config.
            Expected structure:
                datasets:
                    train:
                        _target_: ...
                    val:
                        _target_: ...

    Returns:
        dict[str, Dataset]: mapping of partition name → dataset instance.
    """
    datasets = {}
    for dataset_partition in config.datasets.keys():
        dataset_cfg = config.datasets[dataset_partition]
        dataset = instantiate(dataset_cfg)
        datasets[dataset_partition] = dataset
    return datasets


def get_dataloaders(config, datasets):
    """
    Create DataLoader instances for each dataset partition.

    Args:
        config (DictConfig): Hydra experiment config.
            Expected structure:
                dataloaders:
                    train:
                        _target_: torch.utils.data.DataLoader
                        batch_size: 32
                        ...
                    val:
                        _target_: ...
            Each dataloader config is passed to hydra.utils.instantiate
            with the dataset and collate_fn injected.
        datasets (dict[str, Dataset]): dataset instances from get_datasets.

    Returns:
        dict[str, DataLoader]: mapping of partition name → DataLoader.
    """
    dataloaders = {}
    for dataset_partition, dataset in datasets.items():
        dataloader_cfg = config.dataloaders[dataset_partition]

        if hasattr(dataloader_cfg, "batch_size"):
            assert dataloader_cfg.batch_size <= len(dataset), (
                f"The batch size ({dataloader_cfg.batch_size}) cannot "
                f"be larger than the dataset length ({len(dataset)})"
            )

        partition_dataloader = instantiate(
            dataloader_cfg,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )

        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders
