"""
TEMPLATE DATASET — Base class
=============================
Given a proper index (list[dict]), allows processing different datasets
for the same task in the identical manner.

TEMPLATE ADAPTATION:
  Subclass this and build `self._index` in your __init__.
  See src/datasets/template_dataset.py for a complete example.

  The index is a list[dict]. Each dict represents one training example.
  The keys you choose here must match what collate_fn expects and
  what your model's calculate_loss / forward receives.
"""

import logging
import random

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. Each dict should have the fields expected by
                your collate_fn and model (e.g., features, labels).
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict | None): optional per-key transforms
                applied in preprocess_data.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)

        # TODO: Optionally sort by a key for batch uniformity (e.g., sequence length)
        # if not shuffle_index:
        #     index = self._sort_index(index, key=lambda x: len(x["some_field"]))

        self._index: list[dict] = index
        self.instance_transforms = instance_transforms

        assert self.instance_transforms is None, (
            "Instance transforms are not implemented yet. "
            "Please set instance_transforms to None."
        )

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and return it as a dict.

        NOTE: Key names must be consistent across:
          - dataset __getitem__
          - collate_fn
          - model.calculate_loss / forward

        Args:
            ind (int): index in self._index.

        Returns:
            dict: a single dataset element (possibly transformed).
        """
        instance_data = self._index[ind]
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self):
        return len(self._index)

    def preprocess_data(self, instance_data):
        """
        Apply instance transforms per key.

        Override this in your subclass to convert data to tensors,
        apply augmentations, etc.
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Validate the structure of the index.

        TODO: Add your own assertions for required keys, data types, etc.
        Example:
            for entry in index:
                assert "features" in entry
                assert "label" in entry
        """
        pass  # No validation by default — subclasses should define their own

    @staticmethod
    def _sort_index(index, key):
        """
        Sort the index by a given key function.
        Useful for creating batches with similar characteristics
        (e.g., sequence length) for training stability.

        Args:
            index (list[dict]): index to sort.
            key (callable): function that extracts sort key from an entry.

        Returns:
            list[dict]: sorted index.
        """
        return sorted(index, key=key)

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): index to shuffle and limit.
            limit (int | None): if not None, limit to this many elements.
            shuffle_index (bool): if True, shuffle with seed 42.

        Returns:
            list[dict]: processed index.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
