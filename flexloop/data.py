import random
from itertools import islice

import numpy as np
import jax.numpy as jnp

import torch
from torch.utils.data import DataLoader, Sampler

class InfiniteSampler(Sampler):
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
        worker_init_fn=_worker_init_function,
        **kwargs
      )))
      for name, data in datasets
    ]

  def next(self):
    return {
      name: next(it)
      for name, it in self.data_loaders  
    }

def direct_collate(batch, level=0):
  if isinstance(batch[0], (np.ndarray, jnp.ndarray)):
    return np.concatenate(batch, axis=0)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [direct_collate(samples, level=level+1) for samples in transposed]
  elif isinstance(batch[0], dict):
    return {
      key: direct_collate([
        item[key] for item in batch
      ], level=level+1)
      for key in batch[0]
    }
  else:
    return np.array(batch)

class BatchDistribution(DataDistribution):
  def __init__(self, datasets, accumulate=1, **kwargs):
    self.data_loaders = [
      (name, iter(DataLoader(
        data, sampler=InfiniteSampler(data),
        drop_last=True, collate_fn=direct_collate,
        worker_init_fn=_worker_init_function,
        batch_size=accumulate,
        **kwargs
      )))
      for name, data in datasets
    ]

  def next(self):
    return {
      name: next(it)
      for name, it in self.data_loaders  
    }

class BatchStream:
    def __init__(self, dataset, num_workers=0, timeout=120,
                 accumulate=1, prefetch_factor=2):
        self.dataset = dataset
        self.num_workers = num_workers
        self.timeout = timeout
        self.accumulate = accumulate
        self.prefetch_factor = prefetch_factor
        self.dataloader, self.iter = self.setup_dataloader()

    def __iter__(self):
        while True:
            try:
                yield next(self.iter)
            except TimeoutError:
                self.dataloader, self.iter = self.setup_dataloader()
            except StopIteration:
                self.iter = iter(self.dataloader)

    def setup_dataloader(self):
        dl = DataLoader(
            self.dataset, batch_size=self.accumulate, shuffle=False,
            prefetch_factor=self.prefetch_factor, persistent_workers=True,
            drop_last=False, num_workers=self.num_workers, timeout=self.timeout,
            collate_fn=np_collate, worker_init_fn=_worker_init_function)
        return dl, iter(dl)

class BatchStreamDistribution:
    def __init__(self, datasets, accumulate=1, **kwargs):
        self.data_loaders = [
            (name, iter(BatchStream(data, accumulate=accumulate, **kwargs)))
            for name, data in datasets
        ]

    def next(self):
        return {
            name: next(it)
            for name, it in self.data_loaders  
        }

def np_collate(items):
    return {
        key: np.concatenate([i[key] for i in items], axis=0)
        for key in items[0].keys()
    }

def _worker_init_function(worker_id):
  torch_seed = torch.initial_seed()
  random.seed(torch_seed + worker_id)
  if torch_seed >= 2**30:  # make sure torch_seed + worker_id < 2**32
    torch_seed = torch_seed % 2**30
  np.random.seed(torch_seed + worker_id)
