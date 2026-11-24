# CS336 강의 1: source accounting

## 강의 전체
- 목표: 자원을 계산하는 것, 가지고 있는 자원 내에서 학습의 효율성을 최대화 하는 것

## Tensor
- Tensors are the basic building block for storing everything.
- Almost everything are stored as floating point numbers.
- float32 is 4 bytes.
- mixed precision training : bf16 for parameters, activations, gradients / fp32 for optimizer states

## Compute accounting
- tensor einops: 차원 관련해서 실수하기 쉽다.
- einops_einsum(), einops_reduce(), einops_rearrange()

## tensor_operations_flops
- FLOP: floating-point operation (basic operation like addition or multiplication)

## Linear model
- n points, Each point is d-dimensional, linear model(d->k)
- x(b,d)@w(d,k) --> actual_num_flops = 2*b*d*k (one multiplication & one addition)

## summary
- matrix multiplications dominate: (2 m n p) FLOPs
- FLOP/s depends on hardware and data type
- Model FLOPs utilization (MFU): (actual FLOP/s)/(promised FLOP/s) >= 0.5 면 좋은 편

## arithmetic intensity
- memory-bound: communication time > computation time
- compute-bound: computation time > communication time 
- arithmetic_intensity = flops / bytes
- accelerator_intensity: how much work can the accelerator do per byte transferred?
- memory-bound: arithmetic_intensity < accelerator_intensity
- compute-bound: arithmetic_intensity > accelerator_intensity

## gradients for multilayer perceptrons(MLPs)
- backward pass is 2x more expensive than the forward pass
- forward pass: 2(#data points)(#parameters) FLOPs
- backward pass: 4(#data points)(#parameters) FLOPs
- Total: 6(#data points)(#parameters) FLOPs

## Memory
- parameter_memory = 2*(D*D*L) # (2 bytes for bf16)
- activation_memory = 2*B*D*L # (2 bytes for bf16)
- gradient_memory = 2*parameter_memory # (2 bytes for bf16)
- optimizer_state_memory = 4*(num_parameters) # (4 bytes for fp32) 최적화 때문에 fp32 사용
- num_parameters = D*D*L
- flops = 6*B*num_parameters

## Summary
- Everything is operations on tensors(parameters, gradients, activations, optimizer states, data)
- einops: better way to think about tensor operations
- 6(#data points)(#parameters) FLOPs per training step
- Arithmetic intensity / roofline analysis: compute-bound or memory-bound?
- Matrix multiplications are compute-bound, elementwise operations are memory-bound
- Gradient accumulation, activation checkpointing: reduce memory to use bigger batch sizes