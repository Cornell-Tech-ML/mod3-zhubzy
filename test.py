from minitorch import tensor, TensorBackend, FastOps,CudaOps
FastTensorBackend = TensorBackend(FastOps)
CudaBackedn = TensorBackend(CudaOps)
tensor1 = tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
],FastTensorBackend)


print("Fast ", tensor1.sum(1))
      
tensor2 = tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
],CudaBackedn)
print("Cuda ", tensor2.sum(1))



