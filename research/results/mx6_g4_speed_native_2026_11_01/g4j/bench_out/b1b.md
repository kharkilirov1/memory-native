### bench_train_step -- NVIDIA RTX PRO 6000 Blackwell Server Edition | torch 2.11.0+cu130 | triton 3.6.0 | batch 8 x seq 512 | 24 layers | warmup 30 + 100 timed steps | matmul precision highest | --precision bf16 | adam fused | grad-checkpoint False | sdpa True | lr-sched cosine | preset 1b

| arm | d | heads | total params | block-linear params | bits | ms/step | (ev median) | tok/s | model TFLOPS | vs dense_strong | vs dense | fwd | bwd | opt | update (in bwd) | decode (in fwd) | peak MB | resident MB | state MB | +grad MB | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_strong | 2048 | 32 | 1,210,257,408 | 1,207,959,552 | 96.00 | 130.71 | 130.71 | 31,338 | 227.6 | 1.00x | - | 37.94 | 69.55 | 23.34 | - | - | 27773 | 18476 | 13824.00 | 18432.00 | compile: 1 graph(s), 0 break(s) lr-sched cosine est 25,005 MB |
| lg6_fp4_h2 | 2048 | 32 | 1,210,257,408 | 1,207,959,552 | 6.51 | 109.67 | 109.64 | 37,349 | 271.2 | 0.84x | - | 35.08 | 74.95 | 0.09 | 6.27 | 5.19 | 8326 | 1262 | 937.69 | 1001.69 | native bf16 lg (cublas_bf16) updates/step 96/96 lr-sched cosine est 7,863 MB |
| lg6_fp4_h2+compile | 2048 | 32 | 1,210,257,408 | 1,207,959,552 | 6.51 | 108.40 | 108.39 | 37,785 | 274.4 | 0.83x | - | 35.17 | 73.07 | 0.09 | - | - | 8394 | 1261 | 937.69 | 1001.69 | native bf16 lg (cublas_bf16) compile: 1 graph(s), 0 break(s), lg=op updates/step 96/96 lr-sched cosine est 7,863 MB |
| lg6_fp4_h2_nvfp4fwd | 2048 | 32 | 1,210,257,408 | 1,207,959,552 | 6.51 | 102.61 | 102.60 | 39,918 | 289.9 | 0.79x | - | 27.79 | 74.96 | 0.09 | 6.28 | 2.57 | 8327 | 1263 | 937.69 | 1001.69 | native bf16 lg (nvfp4_scaled_mm) updates/step 96/96 lr-sched cosine est 7,863 MB APPROX: NVFP4 activations |
| lg6_fp4_h2_nvfp4fwd+compile | 2048 | 32 | 1,210,257,408 | 1,207,959,552 | 6.51 | 103.43 | 103.40 | 39,601 | 287.6 | 0.79x | - | 30.88 | 72.22 | 0.13 | - | - | 8332 | 1261 | 937.69 | 1001.69 | NON-FINITE LOSS native bf16 lg (nvfp4_scaled_mm) compile: 1 graph(s), 0 break(s), lg=op updates/step 96/96 lr-sched cosine est 7,863 MB APPROX: NVFP4 activations |

ms/step = wall clock of the synchronized timed window / steps (headline); ev median = median of per-step CUDA events. fwd/bwd/opt = median of the phase steps; update = time inside apply_update_ (tern: the fused-kernel call only), decode = weight()/quantized_weight(); both are subsets of bwd / fwd. peak/resident = CUDA memory above what was allocated before the model (nan on CPU); state/+grad = analytic (bits/8 x block-linear params; +grad = resident gradient bytes), see e1_quality's docstring. model TFLOPS = 6 x total params x tok/s (attention FLOPs not counted). vs dense_strong / vs dense = ms/step ratio (<1 = faster). 'fp32 linears' = under --precision bf16 this arm's custom-autograd block linears run as fp32 islands (no tensor cores), the rest is autocast.

#### dense_strong d=2048: top 10 cuda_kernels of one step (total 152.84 ms in 1165 launches; gemm 84.31 ms, other 23.35 ms, elementwise 23.26 ms, copy/cast 10.48 ms, reduce 6.53 ms, triton 4.78 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 27.161 | 17.8 | 96 |
| `Optimizer.step#AdamW.step` | 23.350 | 15.3 | 1 |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::FusedOpti` | 23.252 | 15.2 | 59 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8>(cutlass_80_tensoro` | 19.567 | 12.8 | 73 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 19.229 | 12.6 | 72 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nt_align8>(cutlass_80_tensoro` | 7.365 | 4.8 | 24 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nn_align8>(cutlass_80_tensoro` | 6.612 | 4.3 | 24 |
| `triton_poi_fused__to_copy_12` | 3.249 | 2.1 | 48 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_6` | 3.214 | 2.1 | 47 |
| `triton_poi_fused__to_copy_t_6` | 3.196 | 2.1 | 48 |

#### lg6_fp4_h2 d=2048: top 10 cuda_kernels of one step (total 107.62 ms in 1766 launches; gemm 83.15 ms, other 10.81 ms, elementwise 6.01 ms, copy/cast 4.60 ms, reduce 3.02 ms, index/gather 0.03 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 26.723 | 24.8 | 96 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8>(cutlass_80_tensoro` | 19.386 | 18.0 | 73 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 19.075 | 17.7 | 72 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nt_align8>(cutlass_80_tensoro` | 7.107 | 6.6 | 24 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nn_align8>(cutlass_80_tensoro` | 6.471 | 6.0 | 24 |
| `_lg_update_kernel` | 5.042 | 4.7 | 96 |
| `_lg_decode_kernel` | 4.765 | 4.4 | 192 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::GeluBackwardCUDAKernelImpl(at::TensorIteratorBas` | 3.171 | 2.9 | 24 |
| `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4,` | 2.765 | 2.6 | 24 |
| `void at::native::(anonymous namespace)::layer_norm_grad_input_kernel_vectorized<float, float, false>(float con` | 1.844 | 1.7 | 49 |

#### lg6_fp4_h2+compile d=2048: top 10 cuda_kernels of one step (total 106.71 ms in 1684 launches; gemm 83.63 ms, other 9.66 ms, reduce 7.30 ms, triton 4.82 ms, copy/cast 0.95 ms, elementwise 0.20 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 26.866 | 25.2 | 96 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8>(cutlass_80_tensoro` | 19.541 | 18.3 | 73 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 19.147 | 17.9 | 72 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nt_align8>(cutlass_80_tensoro` | 7.117 | 6.7 | 24 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nn_align8>(cutlass_80_tensoro` | 6.477 | 6.1 | 24 |
| `_lg_update_kernel` | 5.071 | 4.8 | 96 |
| `_lg_decode_kernel` | 4.509 | 4.2 | 192 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_4` | 3.301 | 3.1 | 47 |
| `triton_poi_fused_gelu_gelu_backward_view_3` | 3.094 | 2.9 | 24 |
| `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4,` | 2.827 | 2.6 | 24 |

#### lg6_fp4_h2_nvfp4fwd d=2048: top 10 cuda_kernels of one step (total 99.66 ms in 2750 launches; gemm 64.19 ms, other 14.66 ms, elementwise 11.39 ms, copy/cast 5.41 ms, reduce 3.97 ms, index/gather 0.03 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8>(cutlass_80_tensoro` | 19.431 | 19.5 | 73 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 19.098 | 19.2 | 72 |
| `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn` | 7.703 | 7.7 | 96 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nt_align8>(cutlass_80_tensoro` | 7.134 | 7.2 | 24 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nn_align8>(cutlass_80_tensoro` | 6.475 | 6.5 | 24 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<c10::BFloat1` | 5.145 | 5.2 | 96 |
| `_lg_update_kernel` | 5.044 | 5.1 | 96 |
| `_quant_act_kernel` | 4.516 | 4.5 | 96 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::GeluBackwardCUDAKernelImpl(at::TensorIteratorBas` | 3.154 | 3.2 | 24 |
| `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4,` | 2.769 | 2.8 | 24 |

#### lg6_fp4_h2_nvfp4fwd+compile d=2048: top 10 cuda_kernels of one step (total 100.52 ms in 2668 launches; gemm 63.00 ms, other 16.10 ms, reduce 9.04 ms, elementwise 5.91 ms, triton 4.65 ms, copy/cast 1.67 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8>(cutlass_80_tensoro` | 19.165 | 19.1 | 73 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 18.708 | 18.6 | 72 |
| `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn` | 7.458 | 7.4 | 96 |
| `_quant_act_kernel` | 7.063 | 7.0 | 96 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nt_align8>(cutlass_80_tensoro` | 7.003 | 7.0 | 24 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nn_align8>(cutlass_80_tensoro` | 6.372 | 6.3 | 24 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<c10::BFloat1` | 5.192 | 5.2 | 96 |
| `_lg_update_kernel` | 5.109 | 5.1 | 96 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_4` | 3.436 | 3.4 | 47 |
| `triton_poi_fused_gelu_gelu_backward_view_3` | 3.068 | 3.1 | 24 |

