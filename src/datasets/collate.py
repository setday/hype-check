"""
TEMPLATE COLLATE
================
Generic collation function for batching.

Iterates over all keys in the first batch element and stacks
everything that can be turned into a tensor.

TODO: Customize for your data types:
  - Add padding logic for variable-length sequences
  - Handle string fields, lists of varying lengths, or nested dicts
  - Handle missing keys gracefully
"""

from typing import List, Dict, Any

import torch


def stack_to_tensor(seq, dtype=None):
    """
    Stack a list of elements into a tensor.

    If elements are already tensors, uses torch.stack.
    Otherwise, creates a new tensor from the list.

    Args:
        seq (list): list of elements (tensors or scalar-convertible).
        dtype (torch.dtype | None): optional cast target dtype.

    Returns:
        torch.Tensor: stacked tensor.
    """
    if torch.is_tensor(seq[0]):
        out = torch.stack(seq, dim=0)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
    else:
        out = torch.tensor(seq, dtype=dtype)
    return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Default collation function: stacks every key found in the batch.

    Args:
        batch: A list of dicts, each from dataset.__getitem__.

    Returns:
        A dict with the same keys as the batch elements, each stacked
        into a batched tensor.

    Example input:
        [{"features": [0.1, 0.2], "label": 3},
         {"features": [0.3, 0.4], "label": 1}]

    Example output:
        {"features": tensor([[0.1, 0.2], [0.3, 0.4]]),
         "label": tensor([3, 1])}
    """
    out = {}
    for key in batch[0].keys():
        try:
            # TODO: Add special handling for variable-length fields
            # e.g., pad sequences to max length in batch
            out[key] = stack_to_tensor([b[key] for b in batch])
        except Exception as e:
            # Skip fields that can't be stacked (e.g., strings, None)
            # TODO: Handle or convert problematic fields explicitly
            pass
    return out
