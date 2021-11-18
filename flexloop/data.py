from itertools import islice

import numpy as np
import jax.numpy as jnp

import torch
from torch.utils.data import DataLoader

class InfiniteSampler:
  def __init__(self, data_set):
    self.size = len(data_set)

  def __iter__(self):
    yield from islice(self.permutation(), 0, None, 1)

  def permutation(self):
    while True:
      yield from torch.randperm(self.size)

def numpy_collate(batch, level=0):
  if isinstance(batch[0], (np.ndarray, jnp.ndarray)):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples, level=level+1) for samples in transposed]
  elif isinstance(batch[0], dict):
    return {
      key: numpy_collate([
        item[key] for item in batch
      ], level=level+1)
      for key in batch[0]
    }
  else:
    return np.array(batch)

class DataDistribution:
  def __init__(self, datasets, **kwargs):
    self.data_loaders = [
      (name, iter(DataLoader(
        data, sampler=InfiniteSampler(data),
        drop_last=True, collate_fn=numpy_collate,
        **kwargs
      )))
      for name, data in datasets
    ]

  def next(self):
    return {
      name: next(it)
      for name, it in self.data_loaders  
    }
