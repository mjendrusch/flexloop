import jax
import jax.numpy as jnp

import haiku as hk

def transform_sigma(function, sigma=1e-2):
    transformed = hk.transform(function)
    def apply(params, key, *args, **kwargs):
        return transformed.apply(params["mu"], key, *args, **kwargs)
    def init(key, *args, **kwargs):
        mus = transformed.init(key, *args, **kwargs)
        sigmas = get_sigma(mus, sigma=sigma)
        return {
            "mus": mus,
            "sigmas": sigmas
        }
    return hk.Transformed(init, apply)

def pgpe_grad(func, has_aux=False, samples=10):
    def inner(params, key, *args, **kwargs):
        def grad_func(x):
            return func(x, key, *args, **kwargs)
        def get_noise(x):
            return jax.random.normal(hk.next_rng_key(), [samples] + list(x.shape), x.dtype)
        mus = params["mu"]
        sigmas = params["sigma"]
        directions = jax.tree_map(get_noise, mus)
        r_plus, aux = jax.vmap(grad_func, in_axes=(0, None, None))(
            jax.tree_map(lambda x, y: x + y, mus, directions, sigmas))
        r_minus, _ = jax.vmap(grad_func, in_axes=(0, None, None))(
            jax.tree_map(lambda x, y: x - y, mus, directions,  sigmas))
        baseline = ((r_plus + r_minus) / 2).mean()
        aux = jax.tree_map(lambda x: x[0], aux)
        s = jax.tree_map(lambda x, y: (x ** 2 - y ** 2) / y, directions, sigmas)
        mu_weight = r_plus - r_minus
        sigma_weight = ((r_plus + r_minus) / 2 - baseline)
        mu_grad = jax.tree_map(lambda x: jnp.einsum("s,s...->...", mu_weight, x), directions) / samples
        sigma_grad = jax.tree_map(lambda x: jnp.einsum("s,s...->...", sigma_weight, x), s) / samples
        grad = {
            "mu": mu_grad,
            "sigma": sigma_grad 
        }
        return (baseline, aux), grad
    return inner

# def pgpe_dgs_grad(func, has_aux=False, samples=10):
#     def inner(params, key, *args, **kwargs):
#         def grad_func(x):
#             return func(x, key, *args, **kwargs)
#         def get_noise(x):
#             return jax.random.normal(hk.next_rng_key(), [samples] + list(x.shape), x.dtype)
#         def tree_norm(x):
#             return jax.tree_util.tree_reduce(lambda a, x: a + (x ** 2).sum(), x)
#         mus = params["mu"]
#         sigmas = params["sigma"]
#         directions = jax.tree_map(get_noise, mus)
#         denominator = jnp.sqrt(tree_norm(directions) + 1e-6)
#         directions = jax.tree_map(lambda x: x / denominator, directions)
#         quadval, aux = quadrature(grad_func, mus, sigmas, directions)
#         r_plus, aux = jax.vmap(grad_func, in_axes=(0, None, None))(
#             jax.tree_map(lambda x, y: x + y, mus, directions, sigmas))
#         r_minus, _ = jax.vmap(grad_func, in_axes=(0, None, None))(
#             jax.tree_map(lambda x, y: x - y, mus, directions,  sigmas))
#         baseline = ((r_plus + r_minus) / 2).mean()
#         aux = jax.tree_map(lambda x: x[0], aux)
#         s = jax.tree_map(lambda x, y: (x ** 2 - y ** 2) / y, directions, sigmas)
#         mu_weight = r_plus - r_minus
#         sigma_weight = ((r_plus + r_minus) / 2 - baseline)
#         mu_grad = jax.tree_map(lambda x: jnp.einsum("s,s...->...", mu_weight, x), directions) / samples
#         sigma_grad = jax.tree_map(lambda x: jnp.einsum("s,s...->...", sigma_weight, x), s) / samples
#         grad = {
#             "mu": mu_grad,
#             "sigma": sigma_grad 
#         }
#         return (baseline, aux), grad
#     return inner

def get_sigma(x, sigma=1e-2, path=[]):
    if isinstance(x, dict):
        result = {}
        for name in x:
            result[name] = get_sigma(
                x[name], sigma=sigma, path=path + [name])
    else:
        result = jnp.full(x.shape, sigma, x.dtype)
    return result
