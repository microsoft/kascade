
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void flashattn_gqa_decode_split_kernel_1(half_t* __restrict__ Output, float* __restrict__ Output_partial, float* __restrict__ glse, int batch);
extern "C" __global__ void flashattn_gqa_decode_split_kernel(half_t* __restrict__ K, __grid_constant__ const CUtensorMap Output_partial_desc, __grid_constant__ const CUtensorMap Q_desc, half_t* __restrict__ V, float* __restrict__ glse, int* __restrict__ head_mapping, int* __restrict__ topk_indices, int batch, int kv_seqlen, int topk);
extern "C" __global__ void __launch_bounds__(128, 1) flashattn_gqa_decode_split_kernel_1(half_t* __restrict__ Output, float* __restrict__ Output_partial, float* __restrict__ glse, int batch) {
  float lse_logsum_local[1];
  float o_accum_local[1];
  float lse_local[8];
  float lse_max_local[1];
  float lse_local_split[1];
  float po_local[1];
  float scale_local[1];
  lse_logsum_local[0] = 0x0p+0f/*0.000000e+00*/;
  o_accum_local[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    lse_local[i] = glse[(((((int64_t)((int)blockIdx.y)) * (int64_t)256) + (((int64_t)((int)blockIdx.x)) * (int64_t)8)) + ((int64_t)i))];
  }
  lse_max_local[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    lse_max_local[0] = max(lse_max_local[0], lse_local[rv]);
  }
  for (int k = 0; k < 7; ++k) {
    lse_local_split[0] = glse[(((((int64_t)((int)blockIdx.y)) * (int64_t)256) + (((int64_t)((int)blockIdx.x)) * (int64_t)8)) + ((int64_t)k))];
    lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[0] - lse_max_local[0])));
  }
  lse_local_split[0] = glse[(((((int64_t)((int)blockIdx.y)) * (int64_t)256) + (((int64_t)((int)blockIdx.x)) * (int64_t)8)) + (int64_t)7)];
  lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[0] - lse_max_local[0])));
  lse_logsum_local[0] = (__log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_1 = 0; k_1 < 8; ++k_1) {
    po_local[0] = Output_partial[((((((int64_t)((int)blockIdx.y)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)1024)) + (((int64_t)k_1) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))];
    lse_local_split[0] = glse[(((((int64_t)((int)blockIdx.y)) * (int64_t)256) + (((int64_t)((int)blockIdx.x)) * (int64_t)8)) + ((int64_t)k_1))];
    scale_local[0] = exp2f((lse_local_split[0] - lse_logsum_local[0]));
    o_accum_local[0] = (o_accum_local[0] + (po_local[0] * scale_local[0]));
  }
  Output[(((((int64_t)((int)blockIdx.y)) * (int64_t)4096) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = ((half_t)o_accum_local[0]);
}

extern "C" __global__ void __launch_bounds__(256, 1) flashattn_gqa_decode_split_kernel(half_t* __restrict__ K, __grid_constant__ const CUtensorMap Output_partial_desc, __grid_constant__ const CUtensorMap Q_desc, half_t* __restrict__ V, float* __restrict__ glse, int* __restrict__ head_mapping, int* __restrict__ topk_indices, int batch, int kv_seqlen, int topk) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[64];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t acc_s_cast[64];
  __shared__ uint64_t mbarrier_mem[9];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Output_partial_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
    mbarrier[6].init(128);
    mbarrier[7].init(128);
    mbarrier[8].init(128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    int head_to_reuse = head_mapping[(int)blockIdx.y];
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[8].expect_transaction(16384);
      tl::tma_load(Q_desc, mbarrier[8], (&(((half_t*)buf_dyn_shmem)[0])), 0, (((int)blockIdx.y) * 4), ((int)blockIdx.x));
      tl::tma_load(Q_desc, mbarrier[8], (&(((half_t*)buf_dyn_shmem)[4096])), 64, (((int)blockIdx.y) * 4), ((int)blockIdx.x));
    }
    mbarrier[8].arrive();
    int condval;
    if ((((int)blockIdx.z) < (((topk + 127) & 1023) >> 7))) {
      condval = 1;
    } else {
      condval = 0;
    }
    int64_t kv_offset = ((((int64_t)((int)blockIdx.x)) * ((int64_t)kv_seqlen)) * (int64_t)1024) + (((int64_t)((int)blockIdx.y)) * (int64_t)128);
    int64_t topk_offset = (((min(((int64_t)((int)blockIdx.z)), (((((int64_t)topk) + (int64_t)127) & (int64_t)1023) >> (int64_t)7)) * (int64_t)128) + ((((((int64_t)topk) + (int64_t)127) >> (int64_t)10) * ((int64_t)((int)blockIdx.z))) * (int64_t)128))) + (((((int64_t)((int)blockIdx.x)) * (int64_t)8) + ((int64_t)head_mapping[((int64_t)((int)blockIdx.y))])) * ((int64_t)topk));
    for (int k = 0; k < (((topk + 127) >> 10) + condval); ++k) {
      mbarrier[((k & 1) + 4)].wait((((k & 3) >> 1) ^ 1));
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        uint4* gmem_ptr = (uint4*)(K + kv_offset + ((((((int64_t)topk_indices[(topk_offset + (((int64_t)k) * (int64_t)128) + ((((((int64_t)i) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))) - (int64_t)8)]) * (int64_t)1024))) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        tl::cp_async_gs<16>((uint4*)(((half_t*)buf_dyn_shmem) + (((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 7680)), gmem_ptr);
      }
      tl::fence_proxy_async();
      tl::mbarrier_cp_async_arrive(mbarrier[(k & 1)]);
      mbarrier[(k & 1)].arrive();
      mbarrier[((k & 1) + 6)].wait((((k & 3) >> 1) ^ 1));
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        uint4* gmem_ptr_1 = (uint4*)(V + kv_offset + ((((((int64_t)topk_indices[(topk_offset + (((int64_t)k) * (int64_t)128) + ((((((int64_t)i_1) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))) - (int64_t)8)]) * (int64_t)1024))) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        tl::cp_async_gs<16>((uint4*)(((half_t*)buf_dyn_shmem) + (((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 40448)), gmem_ptr_1);
      }
      tl::fence_proxy_async();
      tl::mbarrier_cp_async_arrive(mbarrier[((k & 1) + 2)]);
      mbarrier[((k & 1) + 2)].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; ++i_2) {
      *(float2*)(acc_o + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      logsum[i_3] = 0x0p+0f/*0.000000e+00*/;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      scores_max[i_4] = -CUDART_INF_F;
    }
    tl::fence_proxy_async();
    mbarrier[8].wait(0);
    int condval_1;
    if ((((int)blockIdx.z) < (((topk + 127) & 1023) >> 7))) {
      condval_1 = 1;
    } else {
      condval_1 = 0;
    }
    for (int k_1 = 0; k_1 < (((topk + 127) >> 10) + condval_1); ++k_1) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        *(float2*)(acc_s + (i_5 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
      }
      tl::fence_proxy_async();
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      tl::gemm_ss<64, 128, 128, 4, 1, 0, 1, 0, 128, 128, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 8192)])), (&(acc_s[0])));
      mbarrier[((k_1 & 1) + 4)].arrive();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        scores_max_prev[i_7] = scores_max[i_7];
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        scores_max[i_8] = -CUDART_INF_F;
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        #pragma unroll
        for (int rv = 0; rv < 32; ++rv) {
          scores_max[i_9] = max(scores_max[i_9], acc_s[((((rv & 15) * 4) + (i_9 * 2)) + (rv >> 4))]);
        }
        scores_max[i_9] = tl::AllReduce<tl::MaxOp, 4, 1, 0, 128>::run_hopper(scores_max[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_scale[i_10] = exp2f(((scores_max_prev[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 64; ++i_11) {
        acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_11 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_sum[i_12] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
          scores_sum[i_12] = (scores_sum[i_12] + acc_s[((((rv_1 & 15) * 4) + (i_12 * 2)) + (rv_1 >> 4))]);
        }
        scores_sum[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0, 128>::run_hopper(scores_sum[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        logsum[i_13] = ((logsum[i_13] * scores_scale[i_13]) + scores_sum[i_13]);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 32; ++i_14) {
        uint1 __1;
        float2 v_ = *(float2*)(acc_s + (i_14 * 2));
        ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
        ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
        *(uint1*)(acc_s_cast + (i_14 * 2)) = __1;
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 64; ++i_15) {
        acc_o[i_15] = (acc_o[i_15] * scores_scale[((i_15 & 3) >> 1)]);
      }
      tl::fence_proxy_async();
      mbarrier[((k_1 & 1) + 2)].wait(((k_1 & 3) >> 1));
      tl::gemm_rs<64, 128, 128, 4, 1, 0, 0, 0, 128, 128, 0, 0, true>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 40960)])), (&(acc_o[0])));
      mbarrier[((k_1 & 1) + 6)].arrive();
    }
    int condval_3;
    if ((((int)blockIdx.z) < (((topk + 127) & 1023) >> 7))) {
      condval_3 = 1;
    } else {
      condval_3 = 0;
    }
    if (0 < (((topk + 127) >> 10) + condval_3)) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 64; ++i_16) {
        acc_o[i_16] = (acc_o[i_16] / logsum[((i_16 & 3) >> 1)]);
      }
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      logsum[i_17] = (__log2f(logsum[i_17]) + (scores_max[i_17] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      if (((((((int)threadIdx.x) >> 5) * 4) + (i_18 * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        glse[((((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)128)) + (((int64_t)i_18) * (int64_t)64)) + (((int64_t)((int)blockIdx.y)) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)8)) + ((int64_t)((int)blockIdx.z)))] = logsum[i_18];
      }
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_19 = 0; i_19 < 32; ++i_19) {
      if ((((((((int)threadIdx.x) >> 5) * 4) + ((i_19 & 1) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < 1) && (((((((int)threadIdx.x) >> 5) * 4) + ((i_19 & 1) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < 1)) {
        *(float2*)(((float*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_19 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_19 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 32768)) = *(float2*)(acc_o + (i_19 * 2));
      }
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(Output_partial_desc, (&(((float*)buf_dyn_shmem)[32768])), 0, ((int)blockIdx.z), (((int)blockIdx.y) * 4), ((int)blockIdx.x));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_flashattn_gqa_decode_split_kernel = cudaFuncSetAttribute(flashattn_gqa_decode_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 147456);
    if (result_flashattn_gqa_decode_split_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 147456, cudaGetErrorString(result_flashattn_gqa_decode_split_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, int* __restrict__ topk_indices, int* __restrict__ head_mapping, float* __restrict__ glse, float* __restrict__ Output_partial, half_t* __restrict__ Output, int batch, int kv_seqlen, int topk, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap Output_partial_desc;
	CUtensorMapDataType Output_partial_desc_type= (CUtensorMapDataType)7;
	cuuint32_t Output_partial_desc_tensorRank= 4;
	void *Output_partial_desc_globalAddress= Output_partial;
	cuuint64_t Output_partial_desc_globalDim[4]= {128,8,32,batch};
	cuuint64_t Output_partial_desc_globalStride[4]= {4,512,4096,131072};
	cuuint32_t Output_partial_desc_boxDim[4]= {128,1,4,1};
	cuuint32_t Output_partial_desc_elementStrides[4]= {1,1,1,1};
	CUtensorMapInterleave Output_partial_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Output_partial_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion Output_partial_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Output_partial_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Output_partial_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Output_partial_desc, Output_partial_desc_type, Output_partial_desc_tensorRank, Output_partial_desc_globalAddress, Output_partial_desc_globalDim, Output_partial_desc_globalStride + 1, Output_partial_desc_boxDim, Output_partial_desc_elementStrides, Output_partial_desc_interleave, Output_partial_desc_swizzle, Output_partial_desc_l2Promotion, Output_partial_desc_oobFill);

	if (Output_partial_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Output_partial_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Q_desc;
	CUtensorMapDataType Q_desc_type= (CUtensorMapDataType)6;
	cuuint32_t Q_desc_tensorRank= 3;
	void *Q_desc_globalAddress= Q;
	cuuint64_t Q_desc_globalDim[3]= {128,32,batch};
	cuuint64_t Q_desc_globalStride[3]= {2,256,8192};
	cuuint32_t Q_desc_boxDim[3]= {64,64,1};
	cuuint32_t Q_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Q_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Q_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion Q_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Q_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Q_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Q_desc, Q_desc_type, Q_desc_tensorRank, Q_desc_globalAddress, Q_desc_globalDim, Q_desc_globalStride + 1, Q_desc_boxDim, Q_desc_elementStrides, Q_desc_interleave, Q_desc_swizzle, Q_desc_l2Promotion, Q_desc_oobFill);

	if (Q_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Q_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	flashattn_gqa_decode_split_kernel<<<dim3(batch, 8, 8), dim3(256, 1, 1), 147456, stream>>>(K, Output_partial_desc, Q_desc, V, glse, head_mapping, topk_indices, batch, kv_seqlen, topk);
	TILELANG_CHECK_LAST_ERROR("flashattn_gqa_decode_split_kernel");
	flashattn_gqa_decode_split_kernel_1<<<dim3(32, batch, 1), dim3(128, 1, 1), 0, stream>>>(Output, Output_partial, glse, batch);
	TILELANG_CHECK_LAST_ERROR("flashattn_gqa_decode_split_kernel_1");

	return 0;
}

