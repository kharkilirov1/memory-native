### bench_train_step -- NVIDIA RTX PRO 6000 Blackwell Server Edition | torch 2.11.0+cu130 | triton 3.6.0 | batch 32 x seq 128 | 4 layers | warmup 3 + 5 timed steps | matmul precision highest

| arm | d | heads | block-linear params | bits | ms/step | (ev median) | tok/s | vs dense | fwd | bwd | opt | update (in bwd) | decode (in fwd) | peak MB | resident MB | state MB | +grad MB | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_nobias | 128 | 2 | 786,432 | 96.00 | 3.40 | 3.18 | 1,203,919 | 1.00x | 1.12 | 1.79 | 0.40 | - | - | 188 | 31 | 9.00 | 12.00 |  |
| qat_fp4 | 128 | 2 | 786,432 | 96.00 | 9.80 | 9.77 | 418,025 | 2.88x | 7.55 | 2.34 | 0.47 | - | 5.63 | 174 | 15 | 9.00 | 12.00 |  |
| tern_counter | 128 | 2 | 786,432 | 6.56 | 11.92 | 11.88 | 343,700 | 3.50x | 4.75 | 7.52 | 0.33 | 1.08 | - | 166 | 3 | 0.62 | 0.87 |  |
| na_fp4_c8 | 128 | 2 | 786,432 | 8.69 | 25.60 | 25.46 | 160,012 | 7.52x | 7.43 | 19.45 | 0.29 | 16.05 | 3.66 | 171 | 4 | 0.81 | 1.06 |  |
| lg6_fp4_h2 | 128 | 2 | 786,432 | 6.69 | 10.35 | 10.32 | 395,721 | 3.04x | 2.30 | 8.83 | 0.27 | 6.02 | 0.59 | 169 | 6 | 0.63 | 0.88 |  |
| lg8_fp4_h2 | 128 | 2 | 786,432 | 8.69 | 10.31 | 10.29 | 397,224 | 3.03x | 2.33 | 8.81 | 0.27 | 6.04 | 0.59 | 169 | 6 | 0.81 | 1.06 |  |

ms/step = wall clock of the synchronized timed window / steps (headline); ev median = median of per-step CUDA events. fwd/bwd/opt = median of the phase steps; update = time inside apply_update_ (tern: the fused-kernel call only), decode = weight()/quantized_weight(); both are subsets of bwd / fwd. peak/resident = CUDA memory above what was allocated before the model (nan on CPU); state/+grad = analytic (bits/8 x block-linear params; +grad = resident gradient bytes), see e1_quality's docstring.

#### dense_nobias d=128: top 10 cuda_kernels of one step (total 2.08 ms in 284 launches; gemm 0.93 ms, other 0.58 ms, elementwise 0.22 ms, copy/cast 0.18 ms, reduce 0.15 ms, index/gather 0.01 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `Optimizer.step#AdamW.step` | 0.354 | 17.0 | 1 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_128x32_8x5_nn_align1::` | 0.192 | 9.2 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_tn_align1>(cutlass_80_simt_sgemm_64x128_8x5_tn_align1::` | 0.186 | 9.0 | 9 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 0.152 | 7.3 | 9 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(cutlass_80_simt_sgemm_64x64_8x5_nt_align1::Pa` | 0.097 | 4.7 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_128x64_8x5_nt_align1::` | 0.096 | 4.6 | 8 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutlass_80_simt_sgemm_128x64_8x5_nn_align1::` | 0.088 | 4.2 | 8 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 0.086 | 4.1 | 28 |
| `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, BatchedTensor, int, ` | 0.067 | 3.2 | 4 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x64_8x5_tn_align1::Pa` | 0.065 | 3.1 | 8 |

#### qat_fp4 d=128: top 10 cuda_kernels of one step (total 3.18 ms in 884 launches; gemm 0.93 ms, elementwise 0.77 ms, other 0.70 ms, copy/cast 0.39 ms, reduce 0.28 ms, index/gather 0.12 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `Optimizer.step#AdamW.step` | 0.477 | 15.0 | 1 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_128x32_8x5_nn_align1::` | 0.190 | 6.0 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_tn_align1>(cutlass_80_simt_sgemm_64x128_8x5_tn_align1::` | 0.181 | 5.7 | 9 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 0.151 | 4.8 | 9 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 0.128 | 4.0 | 32 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 0.109 | 3.4 | 44 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(cutlass_80_simt_sgemm_64x64_8x5_nt_align1::Pa` | 0.098 | 3.1 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_128x64_8x5_nt_align1::` | 0.096 | 3.0 | 8 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutlass_80_simt_sgemm_128x64_8x5_nn_align1::` | 0.088 | 2.8 | 8 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 0.070 | 2.2 | 32 |

#### tern_counter d=128: top 10 cuda_kernels of one step (total 3.04 ms in 1036 launches; gemm 0.93 ms, copy/cast 0.83 ms, elementwise 0.59 ms, other 0.54 ms, reduce 0.15 ms, index/gather 0.01 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `Optimizer.step#AdamW.step` | 0.259 | 8.5 | 1 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_128x32_8x5_nn_align1::` | 0.189 | 6.2 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_tn_align1>(cutlass_80_simt_sgemm_64x128_8x5_tn_align1::` | 0.181 | 6.0 | 9 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 0.152 | 5.0 | 9 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BUnaryFunctor<int, ` | 0.124 | 4.1 | 96 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<int, int, int, at::native::Bitwise` | 0.099 | 3.3 | 96 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_128x64_8x5_nt_align1::` | 0.098 | 3.2 | 8 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(cutlass_80_simt_sgemm_64x64_8x5_nt_align1::Pa` | 0.097 | 3.2 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutlass_80_simt_sgemm_128x64_8x5_nn_align1::` | 0.087 | 2.9 | 8 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BUnaryFunctor<int, ` | 0.086 | 2.8 | 64 |

#### na_fp4_c8 d=128: top 10 cuda_kernels of one step (total 5.64 ms in 2700 launches; elementwise 2.15 ms, copy/cast 1.26 ms, gemm 0.93 ms, other 0.46 ms, reduce 0.41 ms, index/gather 0.40 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 0.286 | 5.1 | 128 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 0.274 | 4.9 | 144 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<a` | 0.268 | 4.7 | 96 |
| `Optimizer.step#AdamW.step` | 0.237 | 4.2 | 1 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_128x32_8x5_nn_align1::` | 0.191 | 3.4 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_tn_align1>(cutlass_80_simt_sgemm_64x128_8x5_tn_align1::` | 0.181 | 3.2 | 9 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >` | 0.152 | 2.7 | 128 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 0.151 | 2.7 | 9 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<long, long, long, at::native::bina` | 0.144 | 2.5 | 64 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 0.137 | 2.4 | 80 |

#### lg6_fp4_h2 d=128: top 10 cuda_kernels of one step (total 3.30 ms in 1116 launches; gemm 0.93 ms, elementwise 0.77 ms, copy/cast 0.52 ms, other 0.44 ms, index/gather 0.34 ms, reduce 0.28 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<a` | 0.273 | 8.3 | 96 |
| `Optimizer.step#AdamW.step` | 0.222 | 6.7 | 1 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >` | 0.204 | 6.2 | 176 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_128x32_8x5_nn_align1::` | 0.192 | 5.8 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_tn_align1>(cutlass_80_simt_sgemm_64x128_8x5_tn_align1::` | 0.181 | 5.5 | 9 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 0.174 | 5.3 | 80 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 0.151 | 4.6 | 9 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, fl` | 0.130 | 4.0 | 32 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_128x64_8x5_nt_align1::` | 0.097 | 2.9 | 8 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(cutlass_80_simt_sgemm_64x64_8x5_nt_align1::Pa` | 0.097 | 2.9 | 13 |

#### lg8_fp4_h2 d=128: top 10 cuda_kernels of one step (total 3.31 ms in 1116 launches; gemm 0.93 ms, elementwise 0.77 ms, copy/cast 0.52 ms, other 0.46 ms, index/gather 0.34 ms, reduce 0.28 ms)

| kernel | ms | % | calls |
|---|---|---|---|
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<a` | 0.274 | 8.3 | 96 |
| `Optimizer.step#AdamW.step` | 0.235 | 7.1 | 1 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >` | 0.204 | 6.2 | 176 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_128x32_8x5_nn_align1::` | 0.192 | 5.8 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_tn_align1>(cutlass_80_simt_sgemm_64x128_8x5_tn_align1::` | 0.182 | 5.5 | 9 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 0.174 | 5.3 | 80 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 0.152 | 4.6 | 9 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, fl` | 0.131 | 3.9 | 32 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(cutlass_80_simt_sgemm_64x64_8x5_nt_align1::Pa` | 0.098 | 2.9 | 13 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_128x64_8x5_nt_align1::` | 0.098 | 2.9 | 8 |

