# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        
        
        # Process multiple elements per thread for better work distribution
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        
        # Each thread processes multiple elements
        while idx < out_size:
            to_index(idx, out_shape, out_index)
            to_index(idx, in_shape, in_index)
            out[index_to_position(out_index, out_strides)] = fn(
                in_storage[index_to_position(in_index, in_strides)]
            )
            idx += stride

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        
        # Pre-allocate index arrays for each thread
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        
        # Calculate thread indices and stride for processing multiple elements
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        
        # Each thread processes multiple elements in a strided pattern
        while idx < out_size:
            # Convert linear index to dimensional indices
            to_index(idx, out_shape, out_index)
            to_index(idx, a_shape, a_index)
            to_index(idx, b_shape, b_index)
            
            # Calculate positions once
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            
            # Perform operation and store result
            out[out_pos] = fn(
                a_storage[a_pos],
                b_storage[b_pos]
            )
            
            # Move to next element for this thread
            idx += stride

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    if pos == 0:
        sum = 0.0
        for j in range(BLOCK_DIM):
            sum += cache[j]
        out[cuda.blockIdx.x] = sum


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # Shared memory for partial results
        BLOCK_DIM = 1024
        
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        
        # Local array for output indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        
        # Get block and thread positions
        block_pos = cuda.blockIdx.x
        thread_pos = cuda.threadIdx.x
        
        # Initialize cache for this block
        if thread_pos == 0:
            cache[block_pos] = reduce_value
        
        # Ensure all threads have initialized cache
        cuda.syncthreads()
        
        # Process only valid output positions
        if thread_pos < out_size:
            # Convert output position to indices
            to_index(thread_pos, out_shape, out_index)
            
            # Calculate output and input positions
            out_pos = index_to_position(out_index, out_strides)
            initial_input_pos = index_to_position(out_index, a_strides)
            
            # Perform reduction along specified dimension
            for i in range(a_shape[reduce_dim]):
                input_pos = initial_input_pos + i * a_strides[reduce_dim]
                cache[out_pos] = fn(cache[out_pos], a_storage[input_pos])
            
            # Ensure all reductions are complete
            cuda.syncthreads()
            
            # Write final result to output
            if thread_pos == 0 and block_pos < out_size:
                out[block_pos] = cache[block_pos]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    for i in range(0, size, BLOCK_DIM):
        a_shared[ty, tx] = a[ty * size + (i + tx)]
        b_shared[ty, tx] = b[(i + ty) * size + tx]
        cuda.syncthreads()

        for j in range(BLOCK_DIM):
            out[ty * size + tx] += a_shared[ty, j] * b_shared[j, tx]
        cuda.syncthreads()


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    """CUDA tensor matrix multiply function."""
    BLOCK_SIZE = 32  
    
    # Shared memory for storing blocks of input matrices
    a_shared = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=numba.float32)
    b_shared = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=numba.float32)
    
    # Get the thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Calculate the row and column this thread is responsible for
    row = by * BLOCK_SIZE + ty
    col = bx * BLOCK_SIZE + tx
    
    # Initialize the accumulator for this thread
    acc = 0.0
    
    # Calculate number of blocks needed
    n_blocks = (a_shape[-1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # For each block
    for block in range(n_blocks):
        # Load data into shared memory
        a_row = row
        a_col = block * BLOCK_SIZE + tx
        if a_row < a_shape[0] and a_col < a_shape[1]:
            a_idx = a_row * a_strides[0] + a_col * a_strides[1]
            a_shared[ty, tx] = a_storage[a_idx]
        else:
            a_shared[ty, tx] = 0.0
            
        b_row = block * BLOCK_SIZE + ty
        b_col = col
        if b_row < b_shape[0] and b_col < b_shape[1]:
            b_idx = b_row * b_strides[0] + b_col * b_strides[1]
            b_shared[ty, tx] = b_storage[b_idx]
        else:
            b_shared[ty, tx] = 0.0
            
        # Synchronize threads to ensure all data is loaded
        cuda.syncthreads()
        
        # Compute partial dot product for this block
        for k in range(BLOCK_SIZE):
            acc += a_shared[ty, k] * b_shared[k, tx]
            
        # Synchronize before loading next block
        cuda.syncthreads()
    
    # Write result to global memory
    if row < out_shape[0] and col < out_shape[1]:
        out_idx = row * out_strides[0] + col * out_strides[1]
        out[out_idx] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
