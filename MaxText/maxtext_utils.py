"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=bare-except, consider-using-generator
"""Utils that are only interesting to MaxText. """

import jax
import optax
from MaxText import max_utils
from jax.sharding import PartitionSpec as P
from jax.experimental.serialize_executable import deserialize_and_load


import pickle
import functools
from MaxText.input_pipeline import input_pipeline_interface

OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"


def get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for train_step"""
  functional_train = get_functional_train_step(train_step, model, config, state_mesh_shardings)
  functional_train.__name__ = "train_step"
  data_pspec = P(*config.data_sharding)
  data_sharding = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = (state_mesh_shardings, None)  # State, metrics
  static_argnums = ()  # We partial out the static argnums of model and config
  donate_argnums = 0  # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_train_step(train_step, model, config, state_mesh_shardings):
  return functools.partial(train_step, model, config, state_mesh_shardings)


def get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for eval_step"""
  functional_eval = get_functional_eval_step(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  data_pspec = P(*config.data_sharding)
  data_sharding = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = None  # metrics
  static_argnums = ()  # We partial out the static argnums of model, config
  donate_argnums = ()  # state will be kept instead of being donated in eval_step
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_eval_step(eval_step, model, config):
  return functools.partial(eval_step, model, config)


def load_compiled(config, partial_train, state):
  """# Loading a serialized compiled train step function."""

  # Currently partial_train and state  are needed to reconstruct
  # input/output shapes to construct the in_trees and out_trees for load API
  # Parker is working on a serializing these
  def load_serialized_compiled(save_name):
    with open(save_name, "rb") as f:
      serialized_compiled = pickle.load(f)
    return serialized_compiled

  def get_train_input_output_trees(func, input_args, input_kwargs):
    _, in_tree_recreated = jax.tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(func, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

  serialized_compiled = load_serialized_compiled(config.compiled_trainstep_file)
  shaped_batch = input_pipeline_interface.get_shaped_batch(config)
  example_rng = jax.random.PRNGKey(0)
  shaped_input_args = (state, shaped_batch, example_rng)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree)
  return p_train_step


def calculate_tokens_training_per_device(config):
  """Calculate training Tokens per device"""
  return config.max_target_length * config.per_device_batch_size * config.gradient_accumulation_steps


def calculate_gemma2_tflops_training_per_device(config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops):
  """
  Calculate training TFLOP for Gemma2 as in Gemma2 we combine [local_attention, global_attention] into one decoder
  layer and we use sliding window attention in local_attention
  """
  attention_flops = (
      # global attention
      4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
      +
      # local attention
      4
      * config.per_device_batch_size
      * config.max_target_length
      * min(config.sliding_window_size, config.max_target_length)
      * config.num_query_heads
      * config.head_dim
  )
  attention_tflops = attention_flops * config.num_decoder_layers * 3 / 10**12

  # multiply num_decoder_layers by 2 because we combine [local_attention, global_attention] into one decoder layer
  learnable_weight_tflops = (
      ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers * 2 + embedding_flops) * 3 / 10**12
  )

  return attention_tflops, learnable_weight_tflops


def calculate_mla_tflops_per_device(config):
  """Calculate Multi-Head Latent Attention TFLOP"""
  batch_len = config.per_device_batch_size * config.max_target_length
  qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim
  # calculate mla query projection
  if config.q_lora_rank == 0:
    q_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * qk_head_dim_sum
  else:
    # calculate query down and up flops
    q_flops = (
        2 * batch_len * (config.emb_dim * config.q_lora_rank + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum)
    )
  # calculate mla kv projection with down and up flops
  kv_flops = (
      2
      * batch_len
      * (
          config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
          + config.kv_lora_rank * config.num_query_heads * (config.qk_nope_head_dim + config.v_head_dim)
      )
  )
  qkv_flops = q_flops + kv_flops

  attention_flops = 2 * batch_len * config.max_target_length * config.num_query_heads * (qk_head_dim_sum + config.v_head_dim)
  projection_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * config.v_head_dim
  return qkv_flops, attention_flops, projection_flops


def calculate_ffn_mamtul_tflops_per_device(config, mlp_dim):
  """Helper function to calculate matmul TFLOP in ffn based on MLP dimension.

  Applies to:
    - Dense FFN layers (mlp_dim = config.mlp_dim).
    - MoE FFN layers (mlp_dim = config.moe_mlp_dim),
      need to scale by shared_experts or num_experts_per_tok.
  """
  ffn1_flops = (
      2 * config.per_device_batch_size * config.max_target_length * mlp_dim * config.emb_dim * len(config.mlp_activations)
  )
  ffn2_flops = 2 * config.per_device_batch_size * config.max_target_length * mlp_dim * config.emb_dim
  return ffn1_flops + ffn2_flops


def calculate_deepseek_ffn_tflops_per_device(config):
  """Helper function to calculate DeepSeek-style ffn TFLOP"""
  gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
  # Due to the mixed decoder layers, the flops is multiplied by num of layers for both dense and moe
  dense_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * config.first_num_dense_layers
  shared_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.shared_experts
  routed_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
  moe_layers = config.num_decoder_layers - config.first_num_dense_layers
  moe_ffn_flops = (gate_flops + shared_experts_flops + routed_experts_flops) * moe_layers
  total_ffn_flops = dense_ffn_flops + moe_ffn_flops
  return total_ffn_flops


def calculate_tflops_training_per_device(config, log=True):
  """Calculate training TFLOP"""
  # MLP flops
  if config.num_experts > 1:
    # calculation based on dropless implementation
    if config.decoder_block == "deepseek":
      total_ffn_flops = calculate_deepseek_ffn_tflops_per_device(config)
    else:
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
      total_ffn_flops = (
          gate_flops + calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * config.num_experts_per_tok
      )
  else:
    total_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)

  # Attention flops
  if config.attention_type == "mla":
    qkv_flops, attention_flops, projection_flops = calculate_mla_tflops_per_device(config)
  else:
    qkv_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * (config.num_query_heads + 2 * config.num_kv_heads)
        * config.head_dim
    )
    attention_flops = (
        4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
    )
    projection_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * config.num_query_heads
        * config.head_dim
    )

  # Embedding flops
  embedding_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.vocab_size

  # Combine flops with number of decoder layers
  if config.decoder_block == "gemma2":
    attention_tflops, learnable_weight_tflops = calculate_gemma2_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )
  elif config.decoder_block == "deepseek":
    learnable_weight_tflops = (
        (total_ffn_flops + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    attention_tflops = attention_flops * config.num_decoder_layers * 3 / 10**12
  else:
    # multiply by 3 for both feed forward and back propagation flops
    learnable_weight_tflops = (
        ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    # megatron tflops calculation does not account for causality in attention
    attention_tflops = attention_flops * config.num_decoder_layers * 3 / 10**12

  learnable_weight_tflops = learnable_weight_tflops * config.gradient_accumulation_steps
  attention_tflops = attention_tflops * config.gradient_accumulation_steps

  # DPO includes one additional forward pass per gradient accumulation step
  if config.use_dpo:
    reference_model_tflops = learnable_weight_tflops / 3  # additional forward pass
    reference_model_attention_tflops = attention_tflops / 3
    attention_tflops = attention_tflops + reference_model_attention_tflops
  else:
    reference_model_tflops = 0

  total_tflops = learnable_weight_tflops + attention_tflops + reference_model_tflops

  if log:
    print(
        "Per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops/total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops


# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_prefill_tflops_per_device(num_model_parameters, prefill_length, config, log=True):
  """Calculate training TFLOP"""
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length / jax.device_count() / 1e12
  noncasual_attention_flops = (
      4
      * config.num_query_heads
      * config.num_decoder_layers
      * config.head_dim
      * prefill_length**2
      / jax.device_count()
      / 1e12
  )
  causal_attention_tflops = noncasual_attention_flops / 2  # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops

  if log:
    print(
        "Per prefill step per device: \n",
        f"\tTotal TFLOPs: {total_tflops:.2f} \n",
        f"\t\tLearnable weight TFLOPs: {learnable_weight_tflops:.2f} ",
        f"({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n",
        f"\t\tCausal attention TFLOPs: {causal_attention_tflops:.2f} ",
        f"({100 * causal_attention_tflops/total_tflops:.2f})% of Total",
    )
  return total_tflops, learnable_weight_tflops, causal_attention_tflops


def assert_params_sufficiently_sharded(params, mesh, tolerance):
  """Checks whether most params are sharded across sharding axis.

  This function determines whether the majority of parameters  are distributed
  across a specified sharding axes with an acceptable tolerance. It compares the
  current distribution to a scenario where all parameters are fully sharded
  across the 'fsdp', 'fsdp_transpose', 'sequence', and 'tensor' axes.

  Args:
    params: params of the model state
    mesh: mesh constructed from config
    tolerance: float between 0.0 and 1.0 representing the allowed percentage of
    non-sharded parameters.
  Returns:
    bool: True if the majority of parameters are sufficiently sharded
  """
  total_num_params = max_utils.calculate_num_params_from_pytree(params)
  product_num_devices_for_weight_sharding = 1
  for axis in [
      "fsdp",
      "fsdp_transpose",
      "sequence",
      "context",
      "context_autoregressive",
      "tensor",
      "tensor_transpose",
      "tensor_sequence",
      "stage",
      "expert",
  ]:
    product_num_devices_for_weight_sharding *= mesh.shape[axis]
  total_num_params_per_chip = max_utils.calculate_total_params_per_chip(params)
  perfectly_sharded_params_per_chip = total_num_params / product_num_devices_for_weight_sharding
  assert total_num_params_per_chip >= perfectly_sharded_params_per_chip, (
      "Number of parameters per chip must not be less than in the ideal sharded "
      "scenario across `fsdp`, `fsdp_transpose`, `context`, `sequence`, `tensor`, `tensor_transpose`, `tensor_sequence`, `stage`, `expert` axes."
  )
  unsharded_param_perc = total_num_params_per_chip / perfectly_sharded_params_per_chip - 1
  assert unsharded_param_perc < tolerance, (
      f"Number of unsharded parameters exceeds tolerance {tolerance * 100}% "
      f"of total parameters with a value of {unsharded_param_perc * 100}%."
  )


def apply_gradient_clipping(raw_grads, state, clipping_threshold):
  """Applies gradient clipping to raw gradients, with special handing for FLAX fp8 stats.

  Args:
    raw_grads: A pytree of raw gradients.
    state: The current optimizer state.
    clipping_threshold: The gradient clipping threshold.

  Returns:
    A pytree of clipped gradients.
  """
  gradient_clip_transformation = optax.clip_by_global_norm(clipping_threshold)
  if OVERWRITE_WITH_GRADIENT in raw_grads:
    # Scales + Amax History for Delayed Tensor Scaling SHOULD NOT be clipped or affect clipping
    fp8_stats = raw_grads.pop(OVERWRITE_WITH_GRADIENT)
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)
    grads[OVERWRITE_WITH_GRADIENT] = fp8_stats  # pytype: disable=unsupported-operands
    raw_grads[OVERWRITE_WITH_GRADIENT] = fp8_stats  # pytype: disable=unsupported-operands
  else:
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)

  return grads


def get_nested_value(dictionary, nested_key, default=None):
  """
  Retrieves a value from a nested key in a dictionary.

  Args:
      dictionary: The dictionary to search in.
      nested_key: A tuple representing the nested key, e.g., ('level1', 'level2', 'key').
      default: The value to return if the nested key is not found.

  Returns:
      The value associated with the nested key, or the default value if not found.
  """
  current_level = dictionary

  for key in nested_key:
    if not isinstance(current_level, dict) or key not in current_level:
      return default
    current_level = current_level[key]
  return current_level
