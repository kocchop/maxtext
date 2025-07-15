from functools import partial

import jax
import jax.numpy as jnp
from jax import ffi

import transformer_engine.jax as te
import transformer_engine.jax.cpp_extensions as tex
import transformer_engine_jax as tejax

from transformer_engine.jax.quantize import (
    ScaledTensor,
    ScalingMode,
    QuantizerFactory,
    noop_quantizer_set,
)


for _name, _value in tejax.registrations().items():
    ffi.register_ffi_target(_name, _value, platform="CUDA")


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if tejax.get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def grouped_gemm_ffi(
    lhs_list,
    rhs_list,
    lhs_sinv_list,
    rhs_sinv_list,
    *,
    scaling_mode=ScalingMode.NO_SCALING,
):
  assert len(lhs_list) == len(rhs_list)
  out_types = []
  for i in range(len(lhs_list)):
    x = lhs_list[i]
    w = rhs_list[i]

    assert len(x.shape) == 3
    assert len(w.shape) == 3

    # Only support NT layout.
    lhs_to_output_shape = x.shape[:-1]
    rhs_to_output_shape = w.shape[1]
    out_shape = (*lhs_to_output_shape, rhs_to_output_shape)
    out_type = jax.ShapeDtypeStruct(out_shape, jnp.bfloat16)
    out_types.append(out_type)

  workspace_size = get_cublas_workspace_size_bytes() * 4
  workspace_type = jax.ShapeDtypeStruct(shape=(workspace_size,), dtype=jnp.uint8)
  out_types.append(workspace_type)

  num_gemms = len(lhs_list)
  out = jax.ffi.ffi_call("te_grouped_gemm_ffi", tuple(out_types))(
     *lhs_list,
     *rhs_list,
     *lhs_sinv_list,
     *rhs_sinv_list,
     num_gemms=num_gemms,
     scaling_mode=scaling_mode,
     has_bias=0)
  return out[:-1]


def grouped_gemm(
    lhs_list,
    rhs_list
):
    assert len(lhs_list) == len(rhs_list)
    num_gemms = len(lhs_list)
    lhs_list_ = []
    rhs_list_ = []
    lhs_sinv_list_ = []
    rhs_sinv_list_ = []
    for i in range(num_gemms):
        lhs = lhs_list[i]
        rhs = rhs_list[i]



        if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
            scaling_mode = lhs.scaling_mode
            # For ScaledTensors and DELAYED_TENSOR_SCALING, need to handle internal data_layout
            if lhs.scaling_mode.is_tensor_scaling():
                assert not (
                    lhs.data.dtype == jnp.float8_e5m2 and rhs.data.dtype == jnp.float8_e5m2
                ), "FP8 GEMM does not support E5M2 * E5M2"
        else:
            # For jnp.ndarray, only consider contracting_dims, data_layout is always NN
            scaling_mode = ScalingMode.NO_SCALING

        #TODO (Ming Huang): To support MXFP8
        assert scaling_mode in [ScalingMode.NO_SCALING] or \
                scaling_mode.is_tensor_scaling()
        
        if scaling_mode == ScalingMode.NO_SCALING:
            lhs_list_.append(jnp.expand_dims(jnp.squeeze(lhs), axis=0))
            rhs_list_.append(jnp.expand_dims(jnp.squeeze(rhs), axis=0))
            lhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
            rhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
        else:
            lhs_list_.append(jnp.expand_dims(jnp.squeeze(lhs.data), axis=0))
            rhs_list_.append(jnp.expand_dims(jnp.squeeze(rhs.data), axis=0))
        if scaling_mode.is_tensor_scaling():
            lhs_sinv_list_.append(lhs.scale_inv)
            rhs_sinv_list_.append(rhs.scale_inv)

    scaling_mode = scaling_mode.value if scaling_mode.is_tensor_scaling() else scaling_mode
    out_list = grouped_gemm_ffi(
        lhs_list_,
        rhs_list_,
        lhs_sinv_list_,
        rhs_sinv_list_,
        scaling_mode=scaling_mode,
    )

    return out_list


@jax.custom_vjp
def _grouped_dense(x_list, kernel_list, quantizer_set_list):
    output_list, _ = _grouped_dense_fwd_rule(
        x_list, kernel_list, quantizer_set_list
    )
    return output_list


def _grouped_dense_fwd_rule(
    x_list, kernel_list, quantizer_set_list
):
    output_list = []
    x_rowwise_list = []
    x_colwise_list = []
    kernel_colwise_list = []
    kernel_rowwise_list = []
    x_shape_list = []
    kernel_shape_list = []
    if quantizer_set_list is None:
        x_rowwise_list = x_list
        x_colwise_list = x_list
        kernel_colwise_list = kernel_list
        kernel_rowwise_list = kernel_list
        x_shape_list = [x.shape for x in x_list]
        kernel_shape_list = [kernel.shape for kernel in kernel_list]
    else:
        for i in range(len(x_list)):
            q_x = tex.quantize(x_list[i], quantizer_set_list[i].x)
            q_kernel = tex.quantize(kernel_list[i], quantizer_set_list[i].kernel)
            x_rowwise_list.append(q_x.rowwise_tensor)
            x_colwise_list.append(q_x.colwise_tensor)
            kernel_colwise_list.append(q_kernel.colwise_tensor)
            kernel_rowwise_list.append(q_kernel.rowwise_tensor)
            x_shape_list.append(x_rowwise_list[-1].data.shape)
            kernel_shape_list.append(kernel_rowwise_list[-1].data.shape)

    output_list = grouped_gemm(x_rowwise_list, kernel_colwise_list)

    ctx = (
        x_colwise_list,
        kernel_rowwise_list,
        quantizer_set_list,
    )
    return output_list, ctx


def _grouped_dense_bwd_rule(ctx, grad_list):
    (
        colwise_x_list,
        rowwise_kernel_list,
        quantizer_set_list,
    ) = ctx

    group_size = len(grad_list)
    dbias_list = []
    grad_rowwise_list = []
    grad_colwise_list = []

    for i in range(group_size):
        grad = grad_list[i]

        if quantizer_set_list is None:
            casted_grad = grad
            dbias = tex.quantization._jax_dbias(grad)
            grad_rowwise_list.append(grad)
            grad_colwise_list.append(grad)
        else:
            quantizer_set = quantizer_set_list[i]
            casted_grad, dbias = tex.quantize_dbias(
                grad, is_dbias=False, quantizer=quantizer_set.dgrad
            )
            grad_rowwise_list.append(casted_grad.rowwise_tensor)
            grad_colwise_list.append(casted_grad.colwise_tensor)
        dbias_list.append(dbias)

    dgrad_list = grouped_gemm(grad_rowwise_list, rowwise_kernel_list)
    wgrad_list = grouped_gemm(colwise_x_list, grad_colwise_list)

    return list(dgrad_list), list(wgrad_list), quantizer_set_list


_grouped_dense.defvjp(_grouped_dense_fwd_rule, _grouped_dense_bwd_rule)
