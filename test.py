from minitorch import tensor, TensorBackend, FastOps,CudaOps
FastTensorBackend = TensorBackend(FastOps)
CudaBackedn = TensorBackend(CudaOps)
tensor1 = tensor([[[0.00,0.00],[0.00,0.00]],[[0.00,0.00],[0.00,1.00]]],FastTensorBackend)
tensor2 = tensor([[[0.00,0.00],[0.00,1.00]]],FastTensorBackend)

print("FastA ", ((tensor1.view(2,2,2,1))))
print("FastB ", tensor2.view(1,1,2,2))
print("Fast Mul ", ((tensor1.view(2,2,2,1) * tensor2.view(1,1,2,2))))


tensor1 = tensor([[[0.00,0.00],[0.00,0.00]],[[0.00,0.00],[0.00,1.00]]],CudaBackedn)
tensor2 = tensor([[[0.00,0.00],[0.00,1.00]]],CudaBackedn)
print("CudaA ", ((tensor1.view(2,2,2,1))))
print("CudaB ", tensor2.view(1,1,2,2))
print("Cuda Mul ", ((tensor1.view(2,2,2,1) * tensor2.view(1,1,2,2))))