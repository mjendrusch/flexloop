from dataclasses import dataclass
import functools
import time
from typing import Any, Callable
import numpy as np
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
from torch.utils.tensorboard import SummaryWriter
import optax

class Checkpoint:
  def __init__(self, path, save_every=600) -> None:
    self.path = path
    self.last = None
    self.save_every = save_every

  def checkpoint(self, params, step_id):
    with open(f"{self.path}/checkpoint-{step_id}.jax", "wb") as f:
      pickle.dump(params, f)

  def save_aux(self, params, opt_state, aux_state, key, step_id):
    with open(f"{self.path}/save.jax", "wb") as f:
      pickle.dump(dict(
        params=params,
        opt_state=opt_state,
        aux_state=aux_state,
        key=key,
        step_id=step_id
      ), f)

  def save(self, params, opt_state, aux_state, key, step_id):
    if self.last is None:
      self.last = time.time()
    if (time.time() - self.last) > self.save_every:
      self.last = time.time()
      self.save_aux(params, opt_state, aux_state, key, step_id)

  def load(self):
    try:
      with open(f"{self.path}/save.jax", "rb") as f:
        return pickle.load(f)
    except:
      print("could not find save file, starting from random initialization...")
      return None

class Log:
  def __init__(self, path):
    self.writer = SummaryWriter(path)
    self.callbacks = {}

  def add(self, **kwargs):
    for key, val in kwargs.items():
      self.callbacks[key] = val
    return self

  def log(self, name, item, step):
    if isinstance(item, float) or item.ndim == 0:
      self.writer.add_scalar(name, float(item), step)
    if isinstance(item, (np.ndarray, jnp.ndarray, jnp.DeviceArray)):
      if item.ndim == 1:
        item = item.mean()
        self.writer.add_scalar(name, float(item), step)
      if item.ndim in (2, 3):
        self.writer.add_image(name, item, step)
      if item.ndim == 4:
        self.writer.add_images(name, item, step)

  def loggables(self, name, aux, step, path=None):
    path = path or []
    path = path + [name]
    for key, value in aux.items():
      logname = ".".join(path + [key])
      if isinstance(value, dict):
        if len(value) == 2 and "marked" in value:
          del value["marked"]
          kind, val = value.popitem()
          if kind in self.callbacks:
            self.callbacks[kind](self.writer, logname, val, step)
        else:
          self.loggables(key, value, step, path=path)
      else:
        self.log(logname, value, step)

def loggable(name, item):
  return {"marked": 1, name: item}

def batch_to_pmap(batch):
  targets = jax.device_count()
  def _reshape(data):
    return data.reshape(targets, -1, *data.shape[1:])
  return jax.tree_map(_reshape, batch)

def batch_from_pmap(batch):
  def _reshape(data):
    return data.reshape(-1, *data.shape[2:])
  return jax.tree_map(_reshape, batch)

def make_stepgrad_multigpu(stepgrad):
  def inner(params, key, item):
    item = batch_to_pmap(item)
    (loss, aux), grad = jax.pmap(
      stepgrad, in_axes=(None, None, 0), out_axes=0,
      axis_name="i"
    )(params, key, item)
    loss, aux = batch_from_pmap((loss, aux))
    loss = loss.mean()
    grad = jax.tree_map(lambda x: x.mean(axis=0), grad)
    return (loss, aux), grad
  return inner

def cast_float(data, dtype=jnp.bfloat16):
    def _cast_aux(data):
        if isinstance(data, jnp.ndarray) and \
           jnp.issubdtype(data.dtype, jnp.floating):
            return data.astype(dtype)
        return data
    return jax.tree_map(_cast_aux, data)

def cast_gradients(dtype):
  def init_fn(_):
    return optax.EmptyState()

  def update_fn(updates, state, params=None):
    del params
    return cast_float(updates, dtype=dtype), state

  return optax.GradientTransformation(init_fn, update_fn)

def single_update_stepgrad(step, per_item_transform: optax.GradientTransformation):
  stepgrad = jax.value_and_grad(step, 0, has_aux=True)
  def inner(params, key, item):
    transform_state = per_item_transform.init(params)
    def accumulate_fun(item):
        (loss, aux), grad = stepgrad(params, key, item)
        grad, _ = per_item_transform.update(grad, transform_state)
        return (loss, aux), grad
    update = jax.vmap(accumulate_fun, 0, 0)(
      jax.tree_map(lambda x: x[:, None], item))
    (loss, aux) = jax.tree_map(
      lambda x: x.reshape(-1, *x.shape[2:]),
      update[0]
    )
    grad = jax.tree_map(lambda x: x.mean(axis=0), update[1])
    return (loss.mean(axis=0), aux), grad
  return inner

def update_step(step, optimizer: optax.GradientTransformation, accumulate=None, multigpu=False,
                per_item_transform=None):
  stepgrad = jax.value_and_grad(step, 0, has_aux=True)
  if per_item_transform is not None:
    stepgrad = single_update_stepgrad(
      step, per_item_transform=per_item_transform)
  if multigpu:
    stepgrad = make_stepgrad_multigpu(stepgrad)
  def inner(params, opt_state, key, item):
    if accumulate is not None and accumulate > 1:
      def accumulate_fun(item):
        (loss, aux), grad = stepgrad(params, key, item)
        return (loss, aux), grad
      item = jax.tree_map(
        lambda x: x.reshape(accumulate, -1, *x.shape[1:]),
        item
      )
      update = jax.lax.map(accumulate_fun, item)
      (loss, aux), grad = jax.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]),
        update
      )
      loss = loss.mean()
      grad = jax.tree_map(lambda x: x.mean(axis=0), grad)
    else:
      (loss, aux), grad = stepgrad(params, key, item)
    updates, opt_state = optimizer.update(grad, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    aux["gradients"] = loggable("gradients", grad)
    aux["parameters"] = loggable("parameters", params)
    return loss, aux, params, opt_state
  return BakedStep(jax.jit(inner))

class UpdateStep:
  pass

@dataclass
class StagedStep(UpdateStep):
  func: Callable[[Any], Any]

@dataclass
class BakedStep(UpdateStep):
  func: Any
  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)

def staged_update_step(step, optimizer, accumulate=None, multigpu=False,
                       per_item_transform=None):
  @functools.lru_cache(128)
  def inner(aux):
    return update_step(
      functools.partial(step, aux=aux), optimizer,
      accumulate=accumulate, multigpu=multigpu,
      per_item_transform=per_item_transform)
  return StagedStep(inner)

class Step:
  def __init__(self, function):
    self.function = function
    self.loop = None

  def register(self, loop):
    self.loop = loop
    return self

  def __call__(self, *args, **kwargs):
    return self.function(*args, loop=self.loop, **kwargs)

def inert_step(func):
  return Step(func)

class TrainingLoop:
  def __init__(self, log=None, checkpoint=None,
               optimizer=None, max_steps=int(1e7),
               valid_interval=100, checkpoint_interval=1000,
               accumulate=None, multigpu=False,
               per_item_transform=None,
               stage=None,
               node_id=0):
    self.log = log
    self.checkpoint = checkpoint
    self.valid_interval = valid_interval
    self.checkpoint_interval = checkpoint_interval
    self.accumulate = accumulate
    self.multigpu = multigpu
    self.node_id = node_id
    self.optimizer = optimizer
    self.per_item_transform = per_item_transform
    self.step_id = 0
    self.max_steps = max_steps
    self.stage = stage
    self.steps = {}
    self.valid_steps = {}

  def load(self, params, opt_state, aux_state, key):
    archive = self.checkpoint.load()
    if archive is not None:
      self.step_id = archive["step_id"]
      return archive["params"], archive["opt_state"], archive["aux_state"], archive["key"]
    return params, opt_state, aux_state, key

  def bake_steps(self):
    for name, step in self.steps.items():
      if isinstance(step, Step):
        self.steps[name] = step.register(self)
      elif isinstance(step, UpdateStep):
        pass
      else:
        self.steps[name] = update_step(
          step, self.optimizer, accumulate=self.accumulate,
          multigpu=self.multigpu, per_item_transform=self.per_item_transform
        )
    for name, step in self.valid_steps.items():
      self.valid_steps[name] = jax.jit(step)

  def train(self, params, opt_state, key, data, valid=None, aux_state=None, multi_opt=False):
    self.bake_steps()
    aux_state = aux_state or {}
    for idx in range(self.step_id, self.max_steps):
      batch = data.next()
      for name, item in batch.items():
        key, subkey = jax.random.split(key, 2)
        step = self.steps[name]
        if isinstance(step, StagedStep):
          step = step.func(self.stage(self.step_id))
        if isinstance(step, Step):
          if self.node_id == 0:
            params, opt_state, aux_state = step(
              params, opt_state, aux_state, subkey, item
            )
        else:
          if multi_opt:
            loss, aux, params, opt_state[name] = step(params, opt_state[name], subkey, item)
          else:
            loss, aux, params, opt_state = step(params, opt_state, subkey, item)
          if self.node_id == 0:
            self.log.log(name, loss, self.step_id)
            self.log.loggables(name, aux, self.step_id)
      if self.node_id == 0:
        if valid is not None and self.step_id % self.valid_interval == 0:
          batch = valid.next()
          for name, item in batch.items():
            key, subkey = jax.random.split(key, 2)
            loss, aux = self.valid_steps[name](params, subkey, item)
            if self.node_id == 0:
              self.log.log(name, loss, self.step_id)
              self.log.loggables(name, aux, self.step_id)
        if self.step_id % self.checkpoint_interval == 0:
          self.checkpoint.checkpoint(params, self.step_id)
        self.checkpoint.save(params, opt_state, aux_state, subkey, self.step_id)
      self.step_id = idx
    return params
