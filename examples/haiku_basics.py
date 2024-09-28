import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

class MyLinear1(hk.Module):

  def __init__(self, output_size, name=None):
    super().__init__(name=name)
    self.output_size = output_size

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.ones)
    return jnp.dot(x, w) + b
  

def _forward_fn_linear1(x):
  module = MyLinear1(output_size=2)
  return module(x)

forward_linear1 = hk.transform(_forward_fn_linear1)

if __name__ == '__main__':

  print(forward_linear1)
  # dummy_x = jnp.array([[1., 2., 3., 4., 5., 6.]])
  sample_x = jnp.array([[1., 2., 3.]])
  rng_key = jax.random.PRNGKey(42)

  params = forward_linear1.init(rng=rng_key, x=sample_x)
  # finding the shape of params 
  # input is (1,6)
  # w is (6,5)
  # b is (5,)

  print(f'the shape of params is {params["my_linear1"]["w"].shape}')
  print(f'the shape of params is {params["my_linear1"]["b"].shape}')
  
  sample_x = jnp.array([[1., 2., 3.]])
  sample_x_2 = jnp.array([[4., 5., 6.], [7., 8., 9.]])

  output_1 = forward_linear1.apply(params=params, x=sample_x, rng=rng_key)
  # Outputs are identical for given inputs since the forward inference is non-stochastic.
  output_2 = forward_linear1.apply(params=params, x=sample_x, rng=rng_key)

  output_3 = forward_linear1.apply(params=params, x=sample_x_2, rng=rng_key)

  print(f'Output 1 : {output_1}')
  print(f'Output 2 (same as output 1): {output_2}')
  print(f'Output 3 : {output_3}')
  
