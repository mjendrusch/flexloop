import time
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

  def save_aux(self, params, opt_state, key, step_id):
    with open(f"{self.path}/save.jax", "wb") as f:
      pickle.dump(dict(
        params=params,
        opt_state=opt_state,
        key=key,
        step_id=step_id
      ), f)

  def save(self, params, opt_state, key, step_id):
    if self.last is None:
      self.last = time.time()
    if (time.time() - self.last) > self.save_every:
      self.last = time.time()
      self.save_aux(params, opt_state, key, step_id)

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
          self.callbacks[kind](self.writer, logname, val, step)
        else:
          self.loggables(key, value, step, path=path)
      else:
        self.log(logname, value, step)

def loggable(name, item):
  return {"marked": 1, name: item}

def update_step(step, optimizer):
  stepgrad = jax.value_and_grad(step, 0, has_aux=True)
  def inner(params, opt_state, key, item):
    (loss, aux), grad = stepgrad(params, key, item)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, aux, params, opt_state
  return inner

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
               valid_interval=100, checkpoint_interval=1000):
    self.log = log
    self.checkpoint = checkpoint
    self.valid_interval = valid_interval
    self.checkpoint_interval = checkpoint_interval
    self.optimizer = optimizer
    self.step_id = 0
    self.max_steps = max_steps
    self.steps = {}
    self.valid_steps = {}

  def load(self, params, opt_state, key):
    archive = self.checkpoint.load()
    if archive is not None:
      self.step_id = archive["step_id"]
      return archive["params"], archive["opt_state"], archive["key"]
    return params, opt_state, key

  def bake_steps(self):
    for name, step in self.steps.items():
      if isinstance(step, Step):
        self.steps[name] = step.register(self)
      else:
        self.steps[name] = jax.jit(update_step(step, self.optimizer))
    for name, step in self.valid_steps.items():
      self.valid_steps[name] = jax.jit(step)

  def train(self, params, opt_state, key, data, valid=None, aux_state=None):
    self.bake_steps()
    aux_state = aux_state or {}
    for idx in range(self.step_id, self.max_steps):
      batch = data.next()
      for name, item in batch.items():
        if isinstance(self.steps[name], Step):
          params, opt_state, aux_state = self.steps[name](
            params, opt_state, aux_state, key, item
          )
        else:
          loss, aux, params, opt_state = self.steps[name](params, opt_state, key, item)
          self.log.log(name, loss, self.step_id)
          self.log.loggables(name, aux, self.step_id)
      if valid is not None and self.step_id % self.valid_interval == 0:
        batch = valid.next()
        for name, item in batch.items():
          loss, aux = self.valid_steps[name](params, key, item)
          self.log.log(name, loss, self.step_id)
          self.log.loggables(name, aux, self.step_id)
      if self.step_id % self.checkpoint_interval == 0:
        self.checkpoint.checkpoint(params, self.step_id)
      self.checkpoint.save(params, opt_state, key, self.step_id)
      self.step_id = idx
    return params
