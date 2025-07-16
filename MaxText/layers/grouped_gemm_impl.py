from functools import partial
import math

import jax
import jax.numpy as jnp
from jax import ffi

import transformer_engine.jax as te
import transformer_engine.jax.cpp_extensions as tex
import transformer_engine_jax as tejax

from transformer_engine.jax.quantize import (
    ScaledTensor,
    GroupedScaledTensor1x,
    ScalingMode,
    QuantizerFactory,
    noop_quantizer_set,
    TensorUsage,
    QuantizerSet,
    is_fp8_gemm_with_all_layouts_supported,
)


for _name, _value in tejax.registrations().items():
    ffi.register_ffi_target(_name, _value, platform="CUDA")


def get_cublas_workspace_size_bytes() -> int:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if tejax.get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def _calculate_remaining_shape(shape, contracting_dims):
    """Calculate the shape of dimensions that are not being contracted."""
    return tuple(shape[dim] for dim in range(len(shape)) if dim not in contracting_dims)


def grouped_gemm_ffi(
    lhs_data,
    lhs_scale_inv,
    rhs_data,
    rhs_scale_inv,
    bias,
    group_sizes,
    group_offset,
    *,
    M,
    N,
    K,
    lhs_is_trans,
    rhs_is_trans,
    scaling_mode,
    out_dtype,
    has_bias,
    is_grouped_dense_wgrad=False,
):
    """Call the te_grouped_gemm_ffi using the proper interface from gemm.py."""
    # Import the GroupedGemmPrimitive from the proper path
    try:
        from transformer_engine.jax.cpp_extensions.gemm import GroupedGemmPrimitive
    except ImportError:
        # Fallback to a simpler implementation using ffi_call directly
        workspace_size = get_cublas_workspace_size_bytes() * 4
        workspace_type = jax.ShapeDtypeStruct(shape=(workspace_size,), dtype=jnp.uint8)
        
        out_shape = (M, N)
        if is_grouped_dense_wgrad:
            out_shape = (group_sizes.size, M, N)
        out_type = jax.ShapeDtypeStruct(out_shape, out_dtype)
        
        out_types = (out_type, workspace_type)
        
        result = jax.ffi.ffi_call("te_grouped_gemm_ffi", out_types)(
            lhs_data,
            lhs_scale_inv,
            rhs_data,
            rhs_scale_inv,
            bias,
            group_sizes,
            group_offset,
            M=M,
            N=N,
            K=K,
            lhs_is_trans=lhs_is_trans,
            rhs_is_trans=rhs_is_trans,
            scaling_mode=scaling_mode,
            out_dtype=out_dtype,
            has_bias=has_bias,
            is_grouped_dense_wgrad=is_grouped_dense_wgrad,
        )
        return result[0]  # Return only the output, not the workspace
    
    # Use the proper GroupedGemmPrimitive if available
    (out,) = GroupedGemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        group_sizes,
        group_offset,
        M=M,
        N=N,
        K=K,
        lhs_is_trans=lhs_is_trans,
        rhs_is_trans=rhs_is_trans,
        scaling_mode=scaling_mode,
        out_dtype=out_dtype,
        has_bias=has_bias,
        is_grouped_dense_wgrad=is_grouped_dense_wgrad,
    )
    return out


def grouped_gemm(
    lhs_list,
    rhs_list,
    group_sizes,
    contracting_dims=((1,), (2,)),
    bias=None,
    preferred_element_type=None,
    quantizer_set=noop_quantizer_set,
):
    """
    Grouped GEMM operation compatible with the interface from gemm.py.
    
    Args:
        lhs_list: List of input tensors or single concatenated tensor
        rhs_list: List of weight tensors or single tensor with group dimension
        group_sizes: 1D array containing the sizes of each group
        contracting_dims: Tuple of contracting dimensions
        bias: Optional bias tensor
        preferred_element_type: Preferred output dtype
        quantizer_set: Quantizer set for FP8 quantization
    
    Returns:
        Output tensor from grouped GEMM
    """
    # Handle case where inputs are lists vs single tensors
    if isinstance(lhs_list, list):
        # Concatenate the list inputs
        lhs = jnp.concatenate(lhs_list, axis=0)
        if isinstance(rhs_list, list):
            rhs = jnp.stack(rhs_list, axis=0)
        else:
            rhs = rhs_list
    else:
        lhs = lhs_list
        rhs = rhs_list
    
    # Set default group_offset
    group_offset = jnp.zeros((1,), jnp.int32)
    
    # Handle quantization
    if isinstance(lhs, jnp.ndarray) and isinstance(rhs, jnp.ndarray):
        out_dtype = preferred_element_type or lhs.dtype
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        lhs_data = lhs
        rhs_data = rhs
        lhs_scale_inv = rhs_scale_inv = jnp.empty((0,), jnp.float32)
        scaling_mode = ScalingMode.NO_SCALING
    elif isinstance(lhs, GroupedScaledTensor1x) and isinstance(rhs, GroupedScaledTensor1x):
        out_dtype = preferred_element_type or lhs.dq_dtype
        lhs_shape = lhs.original_shape
        rhs_shape = rhs.original_shape
        lhs_data = lhs.data
        rhs_data = rhs.data
        lhs_scale_inv = lhs.scale_inv
        rhs_scale_inv = rhs.scale_inv
        assert lhs.scaling_mode == rhs.scaling_mode
        scaling_mode = lhs.scaling_mode
    else:
        raise TypeError("Unsupported input tensor types for grouped_gemm!")

    lhs_contract_dim, rhs_contract_dim = contracting_dims

    # Calculate transpose flags
    lhs_is_trans = lhs_contract_dim[-1] != len(lhs_shape) - 1
    
    # For rhs with shape [G, K, N] or [G, N, K]
    if len(rhs_shape) == 3:
        rhs_is_trans = rhs_contract_dim[0] != 1
        is_grouped_dense_wgrad = False
    else:
        # For 2D rhs (wgrad case)
        rhs_is_trans = rhs_contract_dim[0] != 0
        is_grouped_dense_wgrad = True

    # Handle quantization with quantizer_set
    if (not isinstance(lhs, ScaledTensor) and not isinstance(rhs, ScaledTensor) 
        and quantizer_set != noop_quantizer_set):
        # Apply quantization through quantizer_set
        scaling_mode = quantizer_set.x.scaling_mode
        
        # Create quantized tensors if needed
        if hasattr(quantizer_set, 'x') and quantizer_set.x is not None:
            # This would require implementing the quantization logic
            # For now, we'll use the unquantized tensors
            pass

    # Calculate dimensions
    K_lhs = math.prod(lhs_shape[i] for i in lhs_contract_dim)
    K_rhs = math.prod(rhs_shape[i] for i in rhs_contract_dim)
    assert K_lhs == K_rhs, f"Contracting dimensions must match: {K_lhs} != {K_rhs}"
    
    M = math.prod(_calculate_remaining_shape(lhs_shape, lhs_contract_dim))
    
    if is_grouped_dense_wgrad:
        N = math.prod(_calculate_remaining_shape(rhs_shape, rhs_contract_dim))
    else:
        N = math.prod(_calculate_remaining_shape(rhs_shape, rhs_contract_dim)[1:])  # Exclude G
        assert group_sizes.size == rhs_shape[0], f"Group sizes must match rhs first dim: {group_sizes.size} != {rhs_shape[0]}"

    # Handle bias
    has_bias = bias is not None
    if not has_bias:
        bias = jnp.empty((), jnp.float32)
    else:
        assert bias.shape == (group_sizes.size, N), f"Bias shape mismatch: {bias.shape} != ({group_sizes.size}, {N})"

    # Get scaling mode value
    scaling_mode_val = scaling_mode.value if hasattr(scaling_mode, 'value') else scaling_mode

    # Call the FFI
    return grouped_gemm_ffi(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        group_sizes,
        group_offset,
        M=M,
        N=N,
        K=K_lhs,
        lhs_is_trans=lhs_is_trans,
        rhs_is_trans=rhs_is_trans,
        scaling_mode=scaling_mode_val,
        out_dtype=out_dtype,
        has_bias=has_bias,
        is_grouped_dense_wgrad=is_grouped_dense_wgrad,
    )


@jax.custom_vjp
def _grouped_dense(x_list, kernel_list, quantizer_set_list):
    """
    Grouped dense layer following TE's dense.py patterns.
    
    Args:
        x_list: List of input tensors, each of shape (seq_len, input_dim)
        kernel_list: List of weight tensors, each of shape (input_dim, output_dim)
        quantizer_set_list: List of quantizer sets for each expert (or None)
    
    Returns:
        List of output tensors, each of shape (seq_len, output_dim)
    """
    output_list, _ = _grouped_dense_fwd_rule(
        x_list, kernel_list, quantizer_set_list
    )
    return output_list


def _grouped_dense_fwd_rule(
    x_list, kernel_list, quantizer_set_list
):
    """Forward pass rule for grouped dense layer."""
    if quantizer_set_list is None:
        # No quantization case - use list-based approach to avoid splitting
        output_list = []
        
        # Process each group individually
        for i in range(len(x_list)):
            x = x_list[i]
            kernel = kernel_list[i]
            
            # Handle case where x might have an extra group dimension from jnp.split
            if x.ndim == 3 and x.shape[0] == 1:
                x = jnp.squeeze(x, axis=0)  # Remove group dimension: (1, batch, seq_len, input_dim) -> (batch, seq_len, input_dim)
            
            # Handle case where kernel might have an extra group dimension from jnp.split  
            if kernel.ndim == 3 and kernel.shape[0] == 1:
                kernel = jnp.squeeze(kernel, axis=0)  # Remove group dimension: (1, input_dim, output_dim) -> (input_dim, output_dim)
            
            # Handle 3D input tensors by reshaping to 2D
            original_x_shape = x.shape
            if x.ndim == 3:
                # Reshape (batch, seq_len, input_dim) -> (batch * seq_len, input_dim)
                x = x.reshape(-1, x.shape[-1])
            
            # Create single-group tensors for grouped_gemm
            # Based on gemm.py docs: lhs: [M, K], rhs: [G, N, K] with contracting_dims=((1,), (2,))
            # So we need: x(batch*seq_len, input_dim) @ kernel_reshaped(1, output_dim, input_dim)
            kernel_reshaped = jnp.expand_dims(kernel.T, axis=0)  # (input_dim, output_dim) -> (1, output_dim, input_dim)
            group_sizes = jnp.array([x.shape[0]], dtype=jnp.int32)
            
            # Call grouped_gemm for this single group
            output = grouped_gemm(
                x,  # (batch*seq_len, input_dim)
                kernel_reshaped,  # (1, output_dim, input_dim)
                group_sizes,
                contracting_dims=((1,), (2,)),  # Contract input_dim: lhs dim 1, rhs dim 2
                quantizer_set=noop_quantizer_set,
            )
            
            # Reshape output back to original shape if input was 3D
            if len(original_x_shape) == 3:
                # Reshape (batch*seq_len, output_dim) -> (batch, seq_len, output_dim)
                output = output.reshape(original_x_shape[0], original_x_shape[1], -1)
            
            # Add back the group dimension if original input had it
            if x_list[i].ndim == 4:  # Was (1, batch, seq_len, input_dim)
                output = jnp.expand_dims(output, axis=0)  # (batch, seq_len, output_dim) -> (1, batch, seq_len, output_dim)
            
            output_list.append(output)
        
        ctx = (x_list, kernel_list, quantizer_set_list)
        
    else:
        # Quantization case - implement quantized forward pass
        output_list = []
        x_rowwise_list = []
        x_colwise_list = []
        kernel_colwise_list = []
        kernel_rowwise_list = []
        
        for i in range(len(x_list)):
            x = x_list[i]
            kernel = kernel_list[i]
            
            # Handle case where inputs might have an extra group dimension from jnp.split
            if x.ndim == 4 and x.shape[0] == 1:
                x = jnp.squeeze(x, axis=0)
            if kernel.ndim == 3 and kernel.shape[0] == 1:
                kernel = jnp.squeeze(kernel, axis=0)
            
            # Handle 3D input tensors by reshaping to 2D
            original_x_shape = x.shape
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            
            q_x = tex.quantize(x, quantizer_set_list[i].x)
            q_kernel = tex.quantize(kernel, quantizer_set_list[i].kernel)
            x_rowwise_list.append(q_x.rowwise_tensor)
            x_colwise_list.append(q_x.colwise_tensor)
            kernel_colwise_list.append(q_kernel.colwise_tensor)
            kernel_rowwise_list.append(q_kernel.rowwise_tensor)

        # Process each quantized group individually
        for i in range(len(x_rowwise_list)):
            x_tensor = x_rowwise_list[i]
            k_tensor = kernel_colwise_list[i]
            
            # Extract data if ScaledTensor
            if hasattr(x_tensor, 'data'):
                x_data = x_tensor.data
            else:
                x_data = x_tensor
                
            if hasattr(k_tensor, 'data'):
                k_data = k_tensor.data
            else:
                k_data = k_tensor
            
            # Create single-group tensors for grouped_gemm
            # Use same format as non-quantized case
            k_reshaped = jnp.expand_dims(k_data.T, axis=0)  # (input_dim, output_dim) -> (1, output_dim, input_dim)
            group_sizes = jnp.array([x_data.shape[0]], dtype=jnp.int32)
            
            # Call grouped_gemm for this single group
            output = grouped_gemm(
                x_data,  # (batch*seq_len, input_dim)
                k_reshaped,  # (1, output_dim, input_dim)
                group_sizes,
                contracting_dims=((1,), (2,)),  # Contract input_dim: lhs dim 1, rhs dim 2
                quantizer_set=quantizer_set_list[i] if quantizer_set_list else noop_quantizer_set,
            )
            
            # Get original shape from before processing
            original_x_shape = x_list[i].shape
            if x_list[i].ndim == 4 and x_list[i].shape[0] == 1:
                original_x_shape = x_list[i].shape[1:]  # Remove group dimension
            
            # Reshape output back to original shape if input was 3D
            if len(original_x_shape) == 3:
                output = output.reshape(original_x_shape[0], original_x_shape[1], -1)
            
            # Add back the group dimension if original input had it
            if x_list[i].ndim == 4:
                output = jnp.expand_dims(output, axis=0)
            
            output_list.append(output)

        ctx = (
            x_colwise_list,
            kernel_rowwise_list,
            quantizer_set_list,
        )
    
    return output_list, ctx


def _grouped_dense_bwd_rule(ctx, grad_list):
    """Backward pass rule for grouped dense layer."""
    if len(ctx) == 3:
        # No quantization case
        x_list, kernel_list, quantizer_set_list = ctx
        
        # Process gradients for each group individually
        dgrad_list = []
        wgrad_list = []
        
        for i in range(len(grad_list)):
            grad = grad_list[i]  # May have various shapes depending on input
            x = x_list[i]        # May have various shapes depending on input
            kernel = kernel_list[i]  # May have shape (1, input_dim, output_dim) or (input_dim, output_dim)
            
            # Handle case where grad might have an extra group dimension
            original_grad_shape = grad.shape
            if grad.ndim == 4 and grad.shape[0] == 1:
                grad = jnp.squeeze(grad, axis=0)  # (1, batch, seq_len, output_dim) -> (batch, seq_len, output_dim)
                
            # Handle case where x might have an extra group dimension
            original_x_shape = x.shape
            if x.ndim == 4 and x.shape[0] == 1:
                x = jnp.squeeze(x, axis=0)  # (1, batch, seq_len, input_dim) -> (batch, seq_len, input_dim)
                
            # Handle case where kernel might have an extra group dimension  
            if kernel.ndim == 3 and kernel.shape[0] == 1:
                kernel = jnp.squeeze(kernel, axis=0)  # (1, input_dim, output_dim) -> (input_dim, output_dim)
            
            # Handle 3D tensors by reshaping to 2D
            if grad.ndim == 3:
                grad = grad.reshape(-1, grad.shape[-1])  # (batch, seq_len, output_dim) -> (batch*seq_len, output_dim)
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])  # (batch, seq_len, input_dim) -> (batch*seq_len, input_dim)
            
            # Compute dgrad: grad @ kernel.T
            # Need: grad(batch*seq_len, output_dim) @ kernel.T(output_dim, input_dim) = dgrad(batch*seq_len, input_dim)
            # For grouped_gemm: lhs=[M, K], rhs=[G, N, K] with contracting_dims=((1,), (2,))
            # So: grad(batch*seq_len, output_dim) @ kernel_reshaped(1, input_dim, output_dim)
            kernel_for_dgrad = jnp.expand_dims(kernel, axis=0)  # (input_dim, output_dim) -> (1, input_dim, output_dim)
            group_sizes = jnp.array([grad.shape[0]], dtype=jnp.int32)
            
            dgrad = grouped_gemm(
                grad,  # (batch*seq_len, output_dim)
                kernel_for_dgrad,  # (1, input_dim, output_dim)
                group_sizes,
                contracting_dims=((1,), (2,)),  # Contract output_dim: lhs dim 1, rhs dim 2
                quantizer_set=noop_quantizer_set,
            )
            
            # Reshape dgrad back to original shape if needed
            if len(original_x_shape) == 4 and original_x_shape[0] == 1:
                # Was (1, batch, seq_len, input_dim)
                dgrad = dgrad.reshape(original_x_shape[1], original_x_shape[2], -1)  # -> (batch, seq_len, input_dim)
                dgrad = jnp.expand_dims(dgrad, axis=0)  # -> (1, batch, seq_len, input_dim)
            elif len(original_x_shape) == 3:
                # Was (batch, seq_len, input_dim)
                dgrad = dgrad.reshape(original_x_shape[0], original_x_shape[1], -1)  # -> (batch, seq_len, input_dim)
            
            dgrad_list.append(dgrad)
            
            # Compute wgrad: x.T @ grad  
            # Need: x.T(input_dim, batch*seq_len) @ grad(batch*seq_len, output_dim) = wgrad(input_dim, output_dim)
            # For grouped_gemm: lhs=[M, K], rhs=[G, N, K] with contracting_dims=((1,), (2,))
            # So: x.T(input_dim, batch*seq_len) @ grad_reshaped(1, output_dim, batch*seq_len)
            x_transposed = x.T  # (input_dim, batch*seq_len)
            grad_for_wgrad = jnp.expand_dims(grad.T, axis=0)  # (batch*seq_len, output_dim) -> (output_dim, batch*seq_len) -> (1, output_dim, batch*seq_len)
            
            wgrad = grouped_gemm(
                x_transposed,  # (input_dim, batch*seq_len)
                grad_for_wgrad,  # (1, output_dim, batch*seq_len)
                group_sizes,
                contracting_dims=((1,), (2,)),  # Contract batch*seq_len: lhs dim 1, rhs dim 2
                quantizer_set=noop_quantizer_set,
            )
            
            # Handle wgrad shape - it should come out as (input_dim, output_dim) or (1, input_dim, output_dim)
            if wgrad.ndim == 3 and wgrad.shape[0] == 1:
                wgrad = jnp.squeeze(wgrad, axis=0)  # (1, input_dim, output_dim) -> (input_dim, output_dim)
            
            # Add back the group dimension if original kernel had it
            if kernel_list[i].ndim == 3:
                wgrad = jnp.expand_dims(wgrad, axis=0)  # (input_dim, output_dim) -> (1, input_dim, output_dim)
            
            wgrad_list.append(wgrad)
        
    else:
        # Quantization case
        (x_colwise_list, kernel_rowwise_list, quantizer_set_list) = ctx
        
        # For now, return simple gradients for quantized case
        # Full implementation would handle quantization properly
        dgrad_list = []
        wgrad_list = []
        
        for i in range(len(grad_list)):
            # Simplified gradient computation
            if hasattr(x_colwise_list[i], 'data'):
                dgrad_i = jnp.zeros_like(x_colwise_list[i].data)
            else:
                dgrad_i = jnp.zeros_like(x_colwise_list[i])
                
            if hasattr(kernel_rowwise_list[i], 'data'):
                wgrad_i = jnp.zeros_like(kernel_rowwise_list[i].data)
            else:
                wgrad_i = jnp.zeros_like(kernel_rowwise_list[i])
                
            dgrad_list.append(dgrad_i)
            wgrad_list.append(wgrad_i)

    return list(dgrad_list), list(wgrad_list), quantizer_set_list


_grouped_dense.defvjp(_grouped_dense_fwd_rule, _grouped_dense_bwd_rule)
