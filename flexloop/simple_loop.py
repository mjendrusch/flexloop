import time
import pickle
import jax.experimental
import jax.experimental.shard_map
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, Mesh, PartitionSpec as PSpec

import haiku as hk
import optax

from collections import namedtuple
from flexloop.loop import cast_float

jax.config.update('jax_threefry_partitionable', True)

State = namedtuple("State", ["key", "step_id", "params", "opt_state", "aux_state"])

def rebatch_call(module, rebatch=1, split_rng=True):
    def inner(data):
        if rebatch > 1:
            if hasattr(module, "config"):
                module.config.rebatch = True
            batched_module = hk.vmap(module, in_axes=0, out_axes=0,
                                     split_rng=split_rng and (not hk.running_init()),
                                     axis_name="rebatch_ax")
            rebatched = rebatch_data(data, rebatch)
            loss, out = batched_module(rebatched)
            loss, out = debatch_output(loss, out)
            # FIXME: this is a band-aid fix for state issues
            if module.config.state and ("_state_update" in out):
                for key, value in out["_state_update"].items():
                    hk.set_state(key, value)
        else:
            loss, out = module(data)
            loss = loss.mean()
        return loss, out
    return inner

def rebatch_call_bysize(module, data_size=1024):
    def inner(data):
        if hk.running_init():
            loss, out = module(data)
            return loss, out
        batched_module = hk.vmap(module, split_rng=(not hk.running_init()))
        data = rebatch_data_bysize(data, data_size)
        loss, out = batched_module(data)
        loss, out = debatch_output(loss, out)
        return loss, out
    return inner

def rebatch_data(data, count):
    return jax.tree_map(lambda x: x.reshape(count, -1, *x.shape[1:]), data)

def rebatch_data_bysize(data, data_size):
    return jax.tree_map(lambda x: x.reshape(-1, data_size, *x.shape[1:]), data)

def debatch_output(loss, out):
    def debatch_inner(x):
        if x.ndim <= 1:
            return x.mean()
        if x.ndim > 1:
            return x[0]
    loss = loss.mean()
    out = jax.tree_map(debatch_inner, out)
    if "losses" in out:
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

def state_sync(state):
    # TODO: synchronize state across devices
    return state

def update_step(step, optimizer, accumulate=1, multigpu=True, with_state=False, nanhunt=False):
    if nanhunt:
        multigpu = False
    stepgrad = make_stepgrad(step, with_state=with_state)
    if accumulate > 1:
        # FIXME: accumulate
        stepgrad = stepgrad_accumulate(stepgrad, batch=accumulate)
    # FIXME: does sharded jit work?
    if multigpu:
        stepgrad = stepgrad_pmap(stepgrad)
    def inner(params, opt_state, key, item):
        if with_state:
            params, state = params
            (loss, (aux, state)), grad = stepgrad(params, state, key, item)
            if multigpu:
                state = state_sync(state)
        else:
            (loss, aux), grad = stepgrad(params, key, item)
        updates, opt_state = optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        aux["total"] = loss.mean()
        if with_state:
            params = (params, state)
        return loss, aux, params, opt_state
    if multigpu:
        result = pmap_jit(inner)
    elif nanhunt:
        result = inner
    else:
        result = jax.jit(inner)
    return result

def valid_step(step, multigpu=True, nanhunt=False, with_state=False):
    if nanhunt:
        multigpu = False
    def inner(params, key, item):
        if with_state:
            (loss, out), _ = step(*params, key, item)
        else:
            loss, out = step(params, key, item)
        out["total"] = loss
        # take the mean over all the GPUs
        if multigpu:
            loss = jax.lax.pmean(loss, axis_name="batch_ax")
            new_aux = {}
            for name, item in out.items():
                if isinstance(item, jnp.ndarray) and item.ndim == 0:
                    item = jax.lax.pmean(item.mean().astype(jnp.float32), axis_name="batch_ax")
                new_aux[name] = item
            out = new_aux
        return out
    if multigpu:
        return valid_pmap_jit(inner)
    elif nanhunt:
        return inner
    return jax.jit(inner)

def make_stepgrad(step, with_state=False):
    if with_state:
        return jax.value_and_grad(_repack_state(step), 0, has_aux=True)
    return jax.value_and_grad(step, 0, has_aux=True)

def _repack_state(step):
    def inner(params, state, key, *args):
        (loss, out), state = step(params, state, key, *args)
        return loss, (out, state)
    return inner

def stepgrad_accumulate(stepgrad, batch=8):
    def _reduce(x):
        if x.ndim == 1:
            return x.mean()
        return x[0]
    def inner(*args):
        params, key, data = args
        def body(carry, xs):
            item, item_key = xs
            (loss, aux), grad = stepgrad(params, item_key, item)
            carry = jax.tree_map(lambda x, y: x + (y - x) / batch, carry, grad)
            return carry, (loss, aux)
        init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        data = rebatch_data(data, batch)
        grad, (loss, aux) = jax.lax.scan(body, init, (data, jax.random.split(key, batch)), unroll=True)
        loss = loss.mean()
        out_aux = jax.tree_map(lambda x: _reduce(x), aux)
        out_aux["total"] = loss
        return (loss, out_aux), grad
    return inner

def stepgrad_end_mean(stepgrad, batch=8):
    def _reduce(x):
        if x.ndim == 1:
            return x.mean()
        return x[0]
    def inner(*args):
        params, key, data = args
        def body(carry, xs):
            item, item_key = xs
            (loss, aux), grad = stepgrad(params, item_key, item)
            return grad, (loss, aux, grad)
        init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        data = rebatch_data(data, batch)
        _, (loss, aux, grad) = jax.lax.scan(body, init, (data, jax.random.split(key, batch)))
        grad = jax.tree_map(lambda x: x.mean(axis=0), grad)
        loss = loss.mean()
        out_aux = jax.tree_map(lambda x: _reduce(x), aux)
        out_aux["total"] = loss
        return (loss, out_aux), grad
    return inner

def stepgrad_map(stepgrad, batch=8):
    def _unbatch(x):
        if x.ndim <= 1:
            return x.mean()
        return x[0]
    def inner(*args):
        params, key, data = args
        data = rebatch_data(data, batch)
        (loss, aux), grad = jax.lax.map(
            lambda x: stepgrad(params, *x),
            (jax.random.split(key, batch), data))
        grad = jax.tree_map(lambda x: x.mean(axis=0), grad)
        loss = loss.mean(axis=0)
        aux = jax.tree_map(_unbatch, aux)
        return (loss, aux), grad
    return inner

def stepgrad_vmap(stepgrad, batch=8):
    def _unbatch(x):
        if x.ndim <= 1:
            return x.mean()
        return x[0]
    def inner(*args):
        params, key, data = args
        data = rebatch_data(data, batch)
        batchgrad = jax.vmap(stepgrad, in_axes=(None, 0, 0), out_axes=(0, 0))
        (loss, aux), grad = batchgrad(params, jax.random.split(key, batch), data)
        grad = jax.tree_map(lambda x: x.mean(axis=0), grad)
        loss = loss.mean(axis=0)
        aux = jax.tree_map(_unbatch, aux)
        return (loss, aux), grad
    return inner

def stepgrad_pmap(stepgrad):
  def inner(*args):
    (loss, aux), grad = stepgrad(*args)
    with_state = False
    if isinstance(aux, tuple):
        aux, state = aux
        with_state = True
    loss = jax.lax.pmean(loss.mean().astype(jnp.float32), axis_name="batch_ax")
    new_aux = {}
    for name, item in aux.items():
        if isinstance(item, jnp.ndarray) and item.ndim == 0:
            print("DO PMEAN")
            item = jax.lax.pmean(item.mean().astype(jnp.float32), axis_name="batch_ax")
        new_aux[name] = item
    aux = new_aux
    grad = jax.lax.pmean(grad, axis_name="batch_ax")
    if with_state:
        aux = (aux, state)
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
      out_axes=None,
      donate_argnums=())(
          params, opt_state, key, data)
  return inner

def shardmap_jit(func):
    mesh = Mesh(mesh_utils.create_device_mesh((NUM_DEVICES,)), ("batch_axis",))
    def inner(params, opt_state, key, data):
        return jax.experimental.shard_map.shard_map(
            func, mesh=mesh,
            in_specs=(None, None, None, PSpec("batch_axis")),
            out_specs=None)(
                params, opt_state, key, data)
    return jax.jit(inner)

def shard_jit(func):
    def shard_in(data, sharding):
        return {
            key: jax.lax.with_sharding_constraint(
                value,
                sharding.reshape(*[NUM_DEVICES] + [1] * (value.ndim - 1)))
            for key, value in data.items()
        }
    sharding = PositionalSharding(mesh_utils.create_device_mesh((NUM_DEVICES,)))
    def inner(params, opt_state, key, data):
        data = shard_in(data, sharding)
        return func(params, opt_state, key, data)
    return jax.jit(inner, out_shardings=None)

def valid_pmap_jit(func):
  def inner(params, key, data):
    data = jax.tree_map(
        lambda x: x.reshape(NUM_DEVICES, -1, *x.shape[1:]),
        data)
    key = jax.random.split(key, NUM_DEVICES)
    return jax.pmap(
      func, axis_name="batch_ax",
      in_axes=(None, 0, 0),
      out_axes=None,
      donate_argnums=())(params, key, data)
  return inner

def log(**loggers):
    def inner(writer, name, item, step, path=None):
        if path is None:
            path = []
        full_name = ".".join(path + [name])
        if isinstance(item, (float, int)) or item.ndim == 0:
            writer.add_scalar(full_name, float(item), step)
        elif isinstance(item, (np.ndarray, jnp.ndarray)):
            if item.ndim == 1:
                item = item.mean()
                writer.add_scalar(full_name, float(item), step)
            if item.ndim in (2, 3):
                writer.add_image(full_name, item, step)
            if item.ndim == 4:
                writer.add_images(full_name, item, step)
        elif isinstance(item, dict):
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
        last_time = None
        for idx in range(step_id, max_steps):
            start_time = time.time()
            key, subkey = jax.random.split(loop_state.key)
            loop_state = loop_state._replace(step_id=idx, key=subkey)
            loop_state, loggables, checkpointables = train_inner(loop_state)
            loop_state = loop_state._replace(key=key)
            for name, loggable in loggables.items():
                logger(writer, name, loggable, idx, path=[])
            if idx % checkpoint_interval == 0:
                for name, item in checkpointables.items():
                    checkpoint(path, name, item, idx)
            last_time = save_loop_state(path, loop_state, last_time, save_interval=save_interval)
            if valid_inner is not None and idx % valid_interval == 0:
                loggables = valid_inner(loop_state)
                for name, loggable in loggables.items():
                    logger(writer, name, loggable, idx, path=["valid"])
            print(f"Step {idx} full iteration: {time.time() - start_time:.3f} s")
    return inner
