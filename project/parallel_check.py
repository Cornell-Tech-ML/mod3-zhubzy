from numba import njit

import minitorch
import minitorch.fast_ops

import numpy as np
# MAP
print("MAP")
tmap = minitorch.fast_ops.tensor_map(njit()(minitorch.operators.id))
out, a = minitorch.zeros((10,)), minitorch.zeros((10,))
tmap(*out.tuple(), *a.tuple())
print(tmap.parallel_diagnostics(level=3))

# ZIP
print("ZIP")
out, a, b = minitorch.zeros((10,)), minitorch.zeros((10,)), minitorch.zeros((10,))
tzip = minitorch.fast_ops.tensor_zip(njit()(minitorch.operators.eq))

tzip(*out.tuple(), *a.tuple(), *b.tuple())
print(tzip.parallel_diagnostics(level=3))

# REDUCE
print("REDUCE")
out, a = minitorch.zeros((1,)), minitorch.zeros((10,))
treduce = minitorch.fast_ops.tensor_reduce(njit()(minitorch.operators.add))

treduce(*out.tuple(), *a.tuple(), 0)
print(treduce.parallel_diagnostics(level=3))


# MM
print("MATRIX MULTIPLY")
out, a, b = (
    minitorch.zeros((3,1, 10, 10)),
    minitorch.zeros((3,1, 10, 20)),
    minitorch.zeros((3, 1, 20, 10)),
)
tmm = minitorch.fast_ops.tensor_matrix_multiply

tmm(*out.tuple(), *a.tuple(), *b.tuple())


def test_4d_matrix_multiply():
    # Test case 1: Basic 4D without broadcasting
    a = np.random.randn(2, 3, 4, 5)  # 2 batch dims, 4x5 matrices
    b = np.random.randn(2, 3, 5, 6)  # 2 batch dims, 5x6 matrices
    
    # NumPy result
    expected = np.matmul(a, b)
    
    # Our implementation
    result = minitorch.fast_ops.tensor_matrix_multiply(
        from_numpy(a),
        from_numpy(b)
    )
    
    np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-7)
    
    
    print("All tests passed!")

test_4d_matrix_multiply()
