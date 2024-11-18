from minitorch import tensor, TensorBackend, FastOps,CudaOps
FastTensorBackend = TensorBackend(FastOps)
CudaBackedn = TensorBackend(CudaOps)
tensor1 = tensor([[[0.00,0.00],[0.00,0.00]],[[0.00,0.00],[0.00,1.00]]],FastTensorBackend)
tensor2 = tensor([[[0.00,0.00],[0.00,1.00]]],FastTensorBackend)

print("Fast ", ((tensor1.view(2,2,2,1) * tensor2.view(1,1,2,2)).sum(2)).view(2,2,2))



tensor1 = tensor([[[0.00,0.00],[0.00,0.00]],[[0.00,0.00],[0.00,1.00]]],CudaBackedn)
tensor2 = tensor([[[0.00,0.00],[0.00,1.00]]],CudaBackedn)
print("Cuda ", ((tensor1.view(2,2,2,1) * tensor2.view(1,1,2,2)).sum(2)).view(2,2,2))