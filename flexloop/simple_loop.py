import time
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from collections import namedtuple
from flexloop.loop import cast_float

State = namedtuple("State", ["key", "step_id", "params", "opt_state", "aux_state"])

def rebatch_call(module, rebatch=1):
    def inner(data):
        if rebatch > 1:
            batched_module = hk.vmap(module, split_rng=(not hk.running_init()))
            rebatched = rebatch_data(data, rebatch)
            loss, out = batched_module(rebatched)
            loss, out = debatch_output(loss, out)
        else:
            loss, out = module(data)
            loss = loss.mean()
        return loss, out
    return inner

def rebatch_data(data, count):
    return jax.tree_map(lambda x: x.reshape(count, -1, *x.shape[1:]), data)

def debatch_output(loss, out):
    def debatch_inner(x):
        if len(x) == 1:
            return x.mean()
        if len(x) > 1:
            return x[0]
    loss = loss.mean()
    out = jax.tree_map(debatch_inner, out)
    out["losses"] = jax.tree_map(lambda x: x.mean(), out["losses"])
    return loss, out

def checkpoint(path, name, item, step_id):
    with open(f"{path}/{name}-{step_id}.jax", "wb") as f:
        pickle.dump(cast_float(item, jnp.float32), f)

def save_loop_state(path, loop_state, last_time, save_interval=600):
    if last_time is None:
        last_time = time.time()
    if (time.time() - last_time) > save_interval:
        last_time = time.time()
        with open(f"{path}/save.jax", "wb") as f:
            pickle.dump(dict(
                params=cast_float(loop_state.params, jnp.float32),
                opt_state=cast_float(loop_state.opt_state, jnp.float32),
                aux_state=cast_float(loop_state.aux_state, jnp.float32),
                key=loop_state.key,
                step_id=loop_state.step_id
            ), f)
    return last_time

def load_loop_state(path):
    try:
        with open(f"{path}/save.jax", "rb") as f:
            res = pickle.load(f)
            return State(res["key"], res["step_id"], res["params"], res["opt_state"], res["aux_state"])
    except:
        print("Could not find save file, starting from random initialization...")
        return None

def update_step(step, optimizer, accumulate=1, multigpu=True):
    stepgrad = make_stepgrad(step)
    if accumulate > 1:
        stepgrad = stepgrad_accumulate(stepgrad, accumulate=accumulate)
    if multigpu:
        stepgrad = stepgrad_pmap(stepgrad)
    def inner(params, opt_state, key, item):
        (loss, aux), grad = stepgrad(params, key, item)
        updates, opt_state = optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        aux["total"] = loss.mean()
        return loss, aux, params, opt_state
    if multigpu:
        result = pmap_jit(inner)
    else:
        result = jax.jit(inner)
    return result

def make_stepgrad(step):
    return jax.value_and_grad(step, 0, has_aux=True)

def stepgrad_accumulate(stepgrad, accumulate=8):
    def inner(*args):
        params, key, data = args
        def body(carry, xs):
            item, key = xs
            (loss, aux), grad = stepgrad(params, key, item)
            carry = jax.tree_map(lambda x, y: x + y, carry, grad)
            return carry, (loss, aux)
        init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        data = jax.tree_map(lambda x: x.reshape(accumulate, -1, *x.shape[1:]), data)
        grad, (loss, aux) = jax.lax.scan(body, init, (data, jax.random.split(key, accumulate)))
        grad = jax.tree_map(lambda x: x / accumulate, grad)
        loss = loss.mean()
        aux = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), aux)
        aux["train"] = loss
        return (loss, aux), grad
    return inner

def stepgrad_pmap(stepgrad):
  def inner(*args):
    (loss, aux), grad = stepgrad(*args)
    loss = jax.lax.pmean(loss.mean().astype(jnp.float32), axis_name="batch_ax")
    new_aux = {}
    for name, item in aux.items():
        if isinstance(item, jnp.ndarray) and item.ndim == 0:
            item = jax.lax.pmean(item.mean().astype(jnp.float32), axis_name="batch_ax")
        new_aux[name] = item
    aux = new_aux
    grad = jax.tree_map(lambda x: jax.lax.pmean(x.astype(jnp.float32), axis_name="batch_ax"), grad)
    return (loss, aux), grad
  return inner

NUM_DEVICES = len(jax.devices())
def pmap_jit(func):
  def inner(params, opt_state, key, data):
    data = jax.tree_map(
        lambda x: x.reshape(NUM_DEVICES, -1, *x.shape[1:]),
        data)
    key = jax.random.split(key, NUM_DEVICES)
    return jax.pmap(
      func, axis_name="batch_ax",
      in_axes=(None, None, 0, 0),
      out_axes=(None, None, None, None),
      donate_argnums=())(params, opt_state, key, data)
  return inner

def log(**loggers):
    def inner(writer, name, item, step, base=""):
        if isinstance(item, (float, int)) or item.ndim == 0:
            writer.add_scalar(name, float(item), step)
        elif isinstance(item, (np.ndarray, jnp.ndarray)):
            if item.ndim == 1:
                item = item.mean()
                writer.add_scalar(name, float(item), step)
            if item.ndim in (2, 3):
                writer.add_image(name, item, step)
            if item.ndim == 4:
                writer.add_images(name, item, step)
        elif isinstance(item, dict):
            full_name = ".".join(base + [name])
            if isinstance(item, dict):
                if len(item) == 2 and "marked" in item:
                    del item["marked"]
                kind, value = item.popitem()
                if kind in loggers:
                    loggers[kind](writer, full_name, value, step)
    return inner

def training(path, train_inner, valid_inner=None,
             max_steps=1_000_000, save_interval=600,
             checkpoint_interval=1000, valid_interval=1000,
             logger=None):
    def inner(writer, loop_state: State):
        step_id = loop_state.step_id
        last_time = 0
        for idx in range(step_id, max_steps):
            key, subkey = jax.random.split(loop_state.key)
            loop_state = loop_state._replace(step_id=idx, key=subkey)
            loop_state, loggables, checkpointables = train_inner(loop_state)
            loop_state = loop_state._replace(key=key)
            for name, loggable in loggables.items():
                logger(writer, name, loggable, idx, base="train")
            if idx % checkpoint_interval == 0:
                for name, item in checkpointables.items():
                    checkpoint(path, name, item, idx)
            save_loop_state(path, loop_state, last_time, save_interval=save_interval)
            if valid_inner is not None and idx % valid_interval == 0:
                loggables = valid_inner(loop_state)
                logger(writer, name, loggable, idx, base="valid")
    return inner
