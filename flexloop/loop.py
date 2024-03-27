from dataclasses import dataclass
import functools
import time
from typing import Any, Callable, NamedTuple
import numpy as np
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
from torch.utils.tensorboard import SummaryWriter
import optax

try:
  import alpa
  def alpa_jit(func):
    return alpa.parallelize(func, donate_argnums=(0, 1, 2), batch_argnums=(3,))
  _HAS_ALPA = True
except:
  _HAS_ALPA = False

NUM_DEVICES = len(jax.devices())
def pmap_jit(func):
  def inner(params, opt_state, key, data):
    data = jax.tree_map(lambda x: x.reshape(NUM_DEVICES, -1, *x.shape[1:]), data)
    key, fixed_key = jax.random.split(key, 2)
    # FIXME
    #data["fixed_key"] = jnp.repeat(fixed_key[None], NUM_DEVICES, axis=0)
    key = jax.random.split(key, NUM_DEVICES)
    return jax.pmap(
      func, axis_name="batch_ax",
      in_axes=(None, None, 0, 0),
      out_axes=(None, None, None, None),
      donate_argnums=(0, 1))(params, opt_state, key, data)
  return inner

def valid_pmap_jit(func):
  def inner(params, key, data):
    data = jax.tree_map(lambda x: x.reshape(NUM_DEVICES, -1, *x.shape[1:]), data)
    key, fixed_key = jax.random.split(key, 2)
    # FIXME
    #data["fixed_key"] = jnp.repeat(fixed_key[None], NUM_DEVICES, axis=0)
    key = jax.random.split(key, NUM_DEVICES)
    return jax.pmap(
      func, axis_name="batch_ax",
      in_axes=(None, 0, 0),
      out_axes=(None, None))(params, key, data)
  return inner

class Checkpoint:
  def __init__(self, path, save_every=600) -> None:
    self.path = path
    self.last = None
    self.save_every = save_every

  def checkpoint(self, params, step_id):
    with open(f"{self.path}/checkpoint-{step_id}.jax", "wb") as f:
      pickle.dump(cast_float(params, jnp.float32), f)

  def save_aux(self, params, opt_state, aux_state, key, step_id):
    with open(f"{self.path}/save.jax", "wb") as f:
      pickle.dump(dict(
        params=cast_float(params, jnp.float32),
        opt_state=cast_float(opt_state, jnp.float32),
        aux_state=cast_float(aux_state, jnp.float32),
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

# class FallbackLog:
#   def __init__(self, path):
#     self.path = path
#     self.callbacks = {}

#   def add(self, **kwargs):
#     for key, val in kwargs.items():
#       self.callbacks[key] = val
#     return self

#   def add_scalar(name, item, step):


#   def log(self, name, item, step):
#     if isinstance(item, float) or item.ndim == 0:
#       self.writer.add_scalar(name, float(item), step)
#     if isinstance(item, (np.ndarray, jnp.ndarray, jnp.DeviceArray)):
#       if item.ndim == 1:
#         item = item.mean()
#         self.writer.add_scalar(name, float(item), step)
#       if item.ndim in (2, 3):
#         self.writer.add_image(name, item, step)
#       if item.ndim == 4:
#         self.writer.add_images(name, item, step)

#   def loggables(self, name, aux, step, path=None):
#     path = path or []
#     path = path + [name]
#     for key, value in aux.items():
#       logname = ".".join(path + [key])
#       if isinstance(value, dict):
#         if len(value) == 2 and "marked" in value:
#           del value["marked"]
#           kind, val = value.popitem()
#           if kind in self.callbacks:
#             self.callbacks[kind](self.writer, logname, val, step)
#         else:
#           self.loggables(key, value, step, path=path)
#       else:
#         self.log(logname, value, step)

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

def pmean_scalars(x):
  if isinstance(x, jnp.ndarray) and \
      jnp.issubdtype(x.dtype, jnp.floating) and \
      x.ndim == 0:
    return jax.lax.pmean(x.astype(jnp.float32), axis_name="batch_ax")

def make_stepgrad_multigpu(stepgrad):
  def inner(*args):
    (loss, aux), grad = stepgrad(*args)
    # average all loss-like scalars over the whole superbatch
    aux = jax.tree_map(pmean_scalars, aux)
    loss = jax.lax.pmean(loss.mean().astype(jnp.float32), axis_name="batch_ax")
    grad = jax.tree_map(lambda x: jax.lax.pmean(x.astype(jnp.float32), axis_name="batch_ax"), grad)
    return (loss, aux), grad
  return inner

def make_valid_multigpu(valid):
  def inner(*args):
    loss, aux = valid(*args)
    loss = jax.lax.pmean(loss.mean().astype(jnp.float32), axis_name="batch_ax")
    aux = jax.tree_map(pmean_scalars, aux)
    return loss, aux
  return inner

def cast_float(data, dtype=jnp.bfloat16):
    def _cast_aux(data):
        if isinstance(data, jnp.ndarray) and \
           jnp.issubdtype(data.dtype, jnp.floating):
          if data.dtype is not dtype:
            data = data.astype(dtype)
          return data
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

def alpa_stepgrad(step, per_item_transform: optax.GradientTransformation):
  stepgrad = alpa.value_and_grad(step, 0, has_aux=True)
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
                per_item_transform=None, with_state=False, jit=True):
  if with_state:
    raw_step = step
    def with_state_step(params, state, key, item):
      (total, aux), state = raw_step(params, state, key, item)
      return total, (state, aux)
    step = with_state_step
  stepgrad = jax.value_and_grad(step, 0, has_aux=True)
  if per_item_transform is not None:
    stepgrad = single_update_stepgrad(
      step, per_item_transform=per_item_transform)
  if (accumulate is not None) and (accumulate > 1):
    stepgrad = accumulate_stepgrad(stepgrad, count=accumulate, with_state=with_state)
  if _HAS_ALPA and multigpu:
    stepgrad = alpa.value_and_grad(step, 0, has_aux=True)
  elif multigpu:
    stepgrad = make_stepgrad_multigpu(stepgrad)
  def inner(params, opt_state, key, item):
    (loss, aux), grad = stepgrad(params, key, item)
    updates, opt_state = optimizer.update(grad, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    aux["gradients"] = loggable("gradients", grad)
    aux["parameters"] = loggable("parameters", params)
    return loss, aux, params, opt_state
  if with_state:
    def inner(ps, opt_state, key, item):
      params, state = ps
      (loss, (state, aux)), grad = stepgrad(params, state, key, item)
      updates, opt_state = optimizer.update(grad, opt_state, params=params)
      params = optax.apply_updates(params, updates)
      aux["gradients"] = loggable("gradients", grad)
      aux["parameters"] = loggable("parameters", params)
      return loss, aux, (params, state), opt_state
  if jit:
    precompile = True
    jitfunction = jax.jit
    if _HAS_ALPA and multigpu:
      precompile = False
      jitfunction = alpa_jit
    elif multigpu:
      precompile = False
      jitfunction = pmap_jit
    result = BakedStep(jitfunction(inner))
    result.initialised = not precompile
    return result
  result = BakedStep(inner)
  result.initialised = True
  return result

def accumulate_stepgrad(stepgrad, count=8, with_state=False):
  def inner(*args):
    params = args[0]
    key = args[-2]
    data = args[-1]
    def body(carry, xs):
      item, key = xs
      (loss, aux), grad = stepgrad(*args[:-2], key, item)
      carry = jax.tree_map(lambda x, y: x + y, carry, grad)
      return carry, (loss, aux)
    init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    data = jax.tree_map(lambda x: x.reshape(count, -1, *x.shape[1:]), data)
    grad, (loss, aux) = jax.lax.scan(body, init, (data, jax.random.split(key, count)))
    grad = jax.tree_map(lambda x: x / count, grad)
    if with_state:
      state, aux = aux
      def slice_state(x):
        return x[0]
      state = jax.tree_map(slice_state, state)
    loss = loss.mean()
    aux = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), aux)
    if with_state:
      aux = (state, aux)
    return (loss, aux), grad
  return inner

class UpdateStep:
  pass

@dataclass
class StagedStep(UpdateStep):
  func: Callable[[Any], Any]

@dataclass
class BakedStep(UpdateStep):
  func: Any
  initialised = False
  def __call__(self, *args, **kwargs):
    if not self.initialised:
      self.initialised = True
      self.func = self.func.lower(*args, **kwargs).compile()
    return self.func(*args, **kwargs)

def tree_map_aux(func, x):
  if isinstance(x, (list, tuple)):
    return [type(x)] + [tree_map_aux(func, i) for i in x]
  if isinstance(x, dict):
    return {n: tree_map_aux(func, i) for n, i in x.items()}
  return func(x)

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
               accumulate=None, multigpu=False, valid_multigpu=False,
               per_item_transform=None, with_state=False,
               stage=None,
               jit=True,
               node_id=0):
    self.log = log
    self.checkpoint = checkpoint
    self.valid_interval = valid_interval
    self.checkpoint_interval = checkpoint_interval
    self.accumulate = accumulate
    self.multigpu = multigpu
    self.valid_multigpu = valid_multigpu
    self.node_id = node_id
    self.optimizer = optimizer
    self.per_item_transform = per_item_transform
    self.step_id = 0
    self.max_steps = max_steps
    self.stage = stage
    self.jit = jit
    self.with_state = with_state
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
        print("ACCUMULATE:", self.accumulate, flush=True)
        optimizer = self.optimizer
        if isinstance(optimizer, dict):
          optimizer = optimizer[name]
        self.steps[name] = update_step(
          step, optimizer, accumulate=self.accumulate,
          multigpu=self.multigpu, per_item_transform=self.per_item_transform,
          jit=self.jit, with_state=self.with_state
        )
    for name, step in self.valid_steps.items():
      if self.valid_multigpu:
        step = make_valid_multigpu(step)
        step = valid_pmap_jit(step)
      else:
        step = jax.jit(step)
      self.valid_steps[name] = step

  # TODO: for some reason the first step runs & compiles twice
  # fix it
  def train(self, params, opt_state, key, data, valid=None, aux_state=None,
            multi_opt=False, batch_step=False):
    self.bake_steps()
    aux_state = aux_state or {}
    for idx in range(self.step_id, self.max_steps):
      t0 = time.time()
      batch = data.next()
      t1 = time.time()
      for name, item in batch.items():
        if batch_step:
          item["step_id"] = jnp.array([self.step_id]) # FIXME: this was a scalar
          # NOTE: to ensure that step_id can be mapped and pmapped over in
          # accumulation & multigpu modes, we replicate it accumulate * NUM_DEVICES times.
          if self.accumulate > 1 or self.multigpu:
            item["step_id"] = jnp.ones((self.accumulate * NUM_DEVICES,), dtype=jnp.int32) * self.step_id
        key, subkey = jax.random.split(key, 2)
        step = self.steps[name]
        if isinstance(step, StagedStep):
          step = step.func(self.stage(self.step_id))
        if isinstance(step, Step):
          if self.node_id == 0:
            update = step(
              params, opt_state, aux_state, subkey, item
            )
            params, opt_state, aux_state = update
        else:
          if multi_opt:
            loss, aux, params, opt_state[name] = step(params, opt_state[name], subkey, item)
            # loss, aux, params, opt_state[name] = update
          else:
            loss, aux, params, opt_state = step(params, opt_state, subkey, item)
            # loss, aux, params, opt_state = update
          if self.node_id == 0:
            self.log.log(name, loss, self.step_id)
            self.log.loggables(name, aux, self.step_id)
        # del update
      t2 = time.time()
      print(f"Step {idx}. Load time {t1 - t0}. Run time {t2 - t1}. Total {t2 - t0}.")
      if self.node_id == 0:
        if valid is not None and self.step_id % self.valid_interval == 0:
          batch = valid.next()
          for name, item in batch.items():
            if batch_step:
              item["step_id"] = jnp.array([self.step_id])
            key, subkey = jax.random.split(key, 2)
            if self.with_state:
              (loss, aux), _ = self.valid_steps[name](params[0], params[1], subkey, item)
            else:
              loss, aux = self.valid_steps[name](params, subkey, item)
            if self.node_id == 0:
              self.log.log(name, loss, self.step_id)
              self.log.loggables(name, aux, self.step_id)
        if self.step_id % self.checkpoint_interval == 0:
          self.checkpoint.checkpoint(params, self.step_id)
        self.checkpoint.save(params, opt_state, aux_state, subkey, self.step_id)
      self.step_id = idx
    return params
