"""
Simple test for the updated grouped_gemm implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from MaxText.layers.grouped_gemm_impl import _grouped_dense, grouped_gemm
from transformer_engine.jax.quantize import noop_quantizer_set


def test_grouped_dense_basic():
    """Test basic functionality of grouped_dense."""
    print("Testing basic grouped_dense functionality...")
    
    # Create test data
    batch_sizes = [4, 6, 3]  # Different batch sizes for different experts
    input_dim = 8
    output_dim = 12
    
    # Create input tensors
    x_list = []
    for batch_size in batch_sizes:
        x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, input_dim))
        x_list.append(x)
    
    # Create weight tensors
    kernel_list = []
    for i in range(len(batch_sizes)):
        key = jax.random.PRNGKey(i + 100)
        kernel = jax.random.normal(key, (input_dim, output_dim))
        kernel_list.append(kernel)
    
    try:
        # Test forward pass without quantization
        outputs = _grouped_dense(x_list, kernel_list, None)
        
        print(f"Forward pass successful!")
        print(f"Number of outputs: {len(outputs)}")
        print(f"Output shapes: {[out.shape for out in outputs]}")
        
        # Verify output shapes
        for i, (x, out) in enumerate(zip(x_list, outputs)):
            expected_shape = (x.shape[0], output_dim)
            assert out.shape == expected_shape, f"Output {i} shape mismatch: {out.shape} != {expected_shape}"
        
        print("✅ Basic test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_grouped_gemm_direct():
    """Test the grouped_gemm function directly."""
    print("\nTesting grouped_gemm function directly...")
    
    try:
        # Create test data
        M = 10  # Total number of tokens
        K = 8   # Input dimension
        N = 12  # Output dimension
        G = 3   # Number of groups
        
        # Create input tensor (M, K)
        lhs = jax.random.normal(jax.random.PRNGKey(42), (M, K))
        
        # Create weight tensor (G, K, N)
        rhs = jax.random.normal(jax.random.PRNGKey(43), (G, K, N))
        
        # Create group sizes
        group_sizes = jnp.array([3, 4, 3], dtype=jnp.int32)  # Sum should equal M
        
        # Test the function
        output = grouped_gemm(
            lhs,
            rhs,
            group_sizes,
            contracting_dims=((1,), (1,)),
            quantizer_set=noop_quantizer_set,
        )
        
        print(f"Direct grouped_gemm test successful!")
        print(f"Input shape: {lhs.shape}")
        print(f"Weight shape: {rhs.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Group sizes: {group_sizes}")
        
        # Verify output shape
        expected_shape = (M, N)
        assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} != {expected_shape}"
        
        print("✅ Direct grouped_gemm test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Direct grouped_gemm test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running grouped_gemm tests...")
    
    # Run tests
    test1_passed = test_grouped_dense_basic()
    test2_passed = test_grouped_gemm_direct()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!") 