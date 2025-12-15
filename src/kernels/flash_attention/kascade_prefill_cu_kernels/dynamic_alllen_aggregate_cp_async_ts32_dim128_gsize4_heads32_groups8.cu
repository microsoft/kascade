	
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

extern "C" __global__ void aggregate_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Q, half_t* __restrict__ aggregate_scores, float* __restrict__ log_sum, int batch, int seq_len);
extern "C" __global__ void __launch_bounds__(384, 1) aggregate_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Q, half_t* __restrict__ aggregate_scores, float* __restrict__ log_sum, int batch, int seq_len) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float log_sum_local[32];
  float acc_s[128];
  float agg_s[4];
  __shared__ uint64_t mbarrier_mem[5];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(256);
    mbarrier[3].init(256);
    mbarrier[4].init(128);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    int tidx = threadIdx.x - 128;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      uint4 condval;
      if ((((((((seq_len + 31) >> 5) * 32) + (i * 2)) + (((int)tidx) >> 6)) - (((int)blockIdx.x) * 32)) < (seq_len + 34))) {
        condval = *(uint4*)(Q + ((((((((((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5) * (int64_t)131072) + (((int64_t)i) * (int64_t)8192)) + ((((int64_t)((int)tidx)) >> (int64_t)6) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)4096)) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + ((((int64_t)((int)tidx)) & (int64_t)63) * (int64_t)8)) - (((int64_t)((int)blockIdx.x)) * (int64_t)131072)) - (int64_t)139264));
      } else {
        condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
      }
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((((int)tidx) & 15) >> 3) * 8192) + (i * 512)) + ((((int)tidx) >> 4) * 64)) + ((((((int)tidx) >> 6) + ((((int)tidx) & 7) >> 2)) & 1) * 32)) + (((((((int)tidx) & 63) >> 5) + ((((int)tidx) & 3) >> 1)) & 1) * 16)) + (((((((int)tidx) & 31) >> 4) + (((int)tidx) & 1)) & 1) * 8)) + 65024)) = condval;
    }
    tl::fence_proxy_async();
    tl::mbarrier_cp_async_arrive(mbarrier[4]);
    mbarrier[4].arrive();
    for (int k = 0; k < ((min(((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)), seq_len) + 223) >> 8); ++k) {
      mbarrier[((k & 1) + 2)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].expect_transaction(65536);
        tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[((k & 1) * 32768)])), 0, ((int)blockIdx.y), (k * 256), ((int)blockIdx.z));
        tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 16384)])), 64, ((int)blockIdx.y), (k * 256), ((int)blockIdx.z));
      }
      mbarrier[(k & 1)].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      float2 condval_1;
      if ((((((((seq_len + 31) >> 5) * 32) + (i_1 * 2)) + ((((int)threadIdx.x) & 3) >> 1)) - (((int)blockIdx.x) * 32)) < (seq_len + 32))) {
        condval_1 = *(float2*)(log_sum + ((((((((((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5) * (int64_t)1024) + (((int64_t)i_1) * (int64_t)64)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)3) >> (int64_t)1) * (int64_t)32)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)32)) + (((int64_t)((int)blockIdx.y)) * (int64_t)4)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)1) * (int64_t)2)) - (((int64_t)((int)blockIdx.x)) * (int64_t)1024)) - (int64_t)1024));
      } else {
        condval_1 = make_float2(CUDART_INF_F, CUDART_INF_F);
      }
      *(float2*)(log_sum_local + (i_1 * 2)) = condval_1;
    }
    tl::fence_proxy_async();
    mbarrier[4].wait(0);
    for (int k_1 = 0; k_1 < ((min(((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)), seq_len) + 223) >> 8); ++k_1) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 128; ++i_2) {
        float condval_2;
        if ((((((((k_1 * 256) + (((i_2 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 32) <= ((((((seq_len + 31) >> 5) * 32) + ((i_2 >> 3) * 2)) + ((((int)threadIdx.x) & 3) >> 1)) - (((int)blockIdx.x) * 32)))) {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_2 = -CUDART_INF_F;
        }
        acc_s[((((((i_2 & 3) >> 1) * 64) + ((i_2 >> 3) * 4)) + ((i_2 & 1) * 2)) + ((i_2 & 7) >> 2))] = condval_2;
      }
      tl::fence_proxy_async();
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      tl::gemm_ss<256, 128, 128, 8, 1, 0, 1, 0, 128, 128, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[((k_1 & 1) * 32768)])), (&(((half_t*)buf_dyn_shmem)[65536])), (&(acc_s[0])));
      mbarrier[((k_1 & 1) + 2)].arrive();
      #pragma unroll
      for (int i_3 = 0; i_3 < 128; ++i_3) {
        acc_s[((((((i_3 & 3) >> 1) * 64) + ((i_3 >> 3) * 4)) + ((i_3 & 1) * 2)) + ((i_3 & 7) >> 2))] = exp2f(((acc_s[((((((i_3 & 3) >> 1) * 64) + ((i_3 >> 3) * 4)) + ((i_3 & 1) * 2)) + ((i_3 & 7) >> 2))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - log_sum_local[(i_3 >> 2)]));
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        agg_s[i_4] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv = 0; rv < 32; ++rv) {
          agg_s[i_4] = (agg_s[i_4] + acc_s[(((((i_4 >> 1) * 64) + ((rv & 15) * 4)) + ((i_4 & 1) * 2)) + (rv >> 4))]);
        }
        agg_s[i_4] = tl::AllReduce<tl::SumOp, 4, 1, 0, 256>::run_hopper(agg_s[i_4]);
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        if (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) <= ((((((k_1 * 256) + ((i_5 >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 32)) {
          agg_s[i_5] = -CUDART_INF_F;
        }
      }
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          if ((((((k_1 * 256) + ((i_6 >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_6 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len) {
            aggregate_scores[((((((((((int64_t)k_1) * (int64_t)256) + ((((int64_t)i_6) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * ((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5)) * (int64_t)8)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * ((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5))) + (((((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5) - ((int64_t)((int)blockIdx.x))) - (int64_t)1) * ((int64_t)seq_len)))] = ((half_t)agg_s[i_6]);
          }
        }
      }
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
    
    cudaError_t result_aggregate_kernel = cudaFuncSetAttribute(aggregate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
    if (result_aggregate_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 163840, cudaGetErrorString(result_aggregate_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, float* __restrict__ log_sum, half_t* __restrict__ aggregate_scores, int batch, int seq_len, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap K_desc;
	CUtensorMapDataType K_desc_type= (CUtensorMapDataType)6;
	cuuint32_t K_desc_tensorRank= 4;
	void *K_desc_globalAddress= K;
	cuuint64_t K_desc_globalDim[4]= {128,8,seq_len,batch};
	cuuint64_t K_desc_globalStride[4]= {2,256,2048,(int64_t)seq_len * 2048};
	cuuint32_t K_desc_boxDim[4]= {64,1,256,1};
	cuuint32_t K_desc_elementStrides[4]= {1,1,1,1};
	CUtensorMapInterleave K_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle K_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion K_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill K_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult K_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &K_desc, K_desc_type, K_desc_tensorRank, K_desc_globalAddress, K_desc_globalDim, K_desc_globalStride + 1, K_desc_boxDim, K_desc_elementStrides, K_desc_interleave, K_desc_swizzle, K_desc_l2Promotion, K_desc_oobFill);

	if (K_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor K_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	aggregate_kernel<<<dim3((seq_len + 31) / 32, 8, batch), dim3(384, 1, 1), 163840, stream>>>(K_desc, Q, aggregate_scores, log_sum, batch, seq_len);
	TILELANG_CHECK_LAST_ERROR("aggregate_kernel");

	return 0;
}

