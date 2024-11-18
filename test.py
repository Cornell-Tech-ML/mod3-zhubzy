from minitorch import tensor, TensorBackend, FastOps,CudaOps
FastTensorBackend = TensorBackend(FastOps)
CudaBackedn = TensorBackend(CudaOps)
tensor1 = tensor([
    [
        [
            [0.0, 0.0],
            [0.0, 0.0]
        ],
        [
            [0.0, 0.0],
            [0.0, 0.0]
        ]
    ],
    [
        [
            [0.0, 0.0],
            [0.0, 0.0]
        ],
        [
            [0.0, 0.0],
            [0.0, 1.0]
        ]
    ]
],FastTensorBackend)


print("Fast ", tensor1.sum(2).view(2,2,2))
      
tensor2 = tensor([
    [
        [
            [0.0, 0.0],
            [0.0, 0.0]
        ],
        [
            [0.0, 0.0],
            [0.0, 0.0]
        ]
    ],
    [
        [
            [0.0, 0.0],
            [0.0, 0.0]
        ],
        [
            [0.0, 0.0],
            [0.0, 1.0]
        ]
    ]
],CudaBackedn)

print("Cuda ", tensor2.sum(2).view(2,2,2))