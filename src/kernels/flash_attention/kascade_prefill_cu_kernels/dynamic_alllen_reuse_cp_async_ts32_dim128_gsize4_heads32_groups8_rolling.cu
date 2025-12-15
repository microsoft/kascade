	
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

extern "C" __global__ void main_kernel(half_t* __restrict__ K, __grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, half_t* __restrict__ Q, half_t* __restrict__ V, __grid_constant__ const CUtensorMap V_desc, int* __restrict__ head_mapping, int* __restrict__ topk_indices, int batch, int max_topk_num, int seq_len, float topk_percent);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(half_t* __restrict__ K, __grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, half_t* __restrict__ Q, half_t* __restrict__ V, __grid_constant__ const CUtensorMap V_desc, int* __restrict__ head_mapping, int* __restrict__ topk_indices, int batch, int max_topk_num, int seq_len, float topk_percent) {
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
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
    mbarrier[6].init(256);
    mbarrier[7].init(256);
    mbarrier[8].init(128);
  }
  __syncthreads();
  int topk_num;
  if ((128 < seq_len)) {
    topk_num = ((max(((int)((topk_percent / 100) * ((((seq_len + 31) >> 5) - ((int)blockIdx.x)) - 1) * 32)), 128) + 127) / 128) * 128;
  } else {
    topk_num = max_topk_num;
  }
              
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    int tidx = ((int)threadIdx.x) - 128;
    int head_to_reuse = head_mapping[(int)blockIdx.y];
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
    tl::mbarrier_cp_async_arrive(mbarrier[8]);
    mbarrier[8].arrive();
    int condval_1;
    if ((((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) <= (topk_num + 32))) {
      condval_1 = ((((((seq_len + 31) >> 5) * 32) + 127) - (((int)blockIdx.x) * 32)) >> 7);
    } else {
      condval_1 = (((topk_num + 127) >> 7) + 1);
    }
    for (int k = 0; k < condval_1; ++k) {
      mbarrier[((k & 1) + 4)].wait((((k & 3) >> 1) ^ 1));
      if (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) <= (topk_num + 32)) {
        if (tl::tl_shuffle_elect<128>()) {
          mbarrier[(k & 1)].expect_transaction(32768);
          tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[((k & 1) * 16384)])), 0, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
          tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 8192)])), 64, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
        }
      } else {
        if (((k * 128) + 128) <= topk_num) {
          #pragma unroll
          for (int i_1 = 0; i_1 < 16; ++i_1) {
            tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((k & 1) * 32768) + (((((int)tidx) & 15) >> 3) * 16384)) + (i_1 * 1024)) + ((((int)tidx) >> 4) * 128)) + ((((((int)tidx) >> 6) + ((((int)tidx) & 7) >> 2)) & 1) * 64)) + (((((((int)tidx) & 63) >> 5) + ((((int)tidx) & 3) >> 1)) & 1) * 32)) + (((((((int)tidx) & 31) >> 4) + (((int)tidx) & 1)) & 1) * 16)) - 1024), K+((((((int64_t)topk_indices[(((((((((int64_t)k) * (int64_t)128) + (((((int64_t)((int)blockIdx.z)) * ((int64_t)max_topk_num)) * ((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5)) * (int64_t)8)) + (((int64_t)i_1) * (int64_t)8)) + (((int64_t)((int)tidx)) >> (int64_t)4)) + ((((int64_t)head_to_reuse) * ((int64_t)max_topk_num)) * ((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5))) + (((((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5) - ((int64_t)((int)blockIdx.x))) - (int64_t)1) * ((int64_t)max_topk_num))) - (int64_t)8)]) * (int64_t)1024) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)tidx)) & (int64_t)15) * (int64_t)8)));
          }
        } else {
            if (tl::tl_shuffle_elect<128>()) {
              mbarrier[(k & 1)].expect_transaction(32768);
              tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[((k & 1) * 16384)])), 0, ((int)blockIdx.y), (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) - 32), ((int)blockIdx.z));
              tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 8192)])), 64, ((int)blockIdx.y), (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) - 32), ((int)blockIdx.z));
            }
        }
      }
      tl::mbarrier_cp_async_arrive(mbarrier[(k & 1)]);
      mbarrier[(k & 1)].arrive();
      mbarrier[((k & 1) + 6)].wait((((k & 3) >> 1) ^ 1));
      if (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) <= (topk_num + 32)) {
        if (tl::tl_shuffle_elect<128>()) {
          mbarrier[((k & 1) + 2)].expect_transaction(32768);
          tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 32768)])), 0, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
          tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 40960)])), 64, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
        }
      } else {
        if (((k * 128) + 128) <= topk_num) {
          #pragma unroll
          for (int i_3 = 0; i_3 < 16; ++i_3) {
            tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((k & 1) * 32768) + (((((int)tidx) & 15) >> 3) * 16384)) + (i_3 * 1024)) + ((((int)tidx) >> 4) * 128)) + ((((((int)tidx) >> 6) + ((((int)tidx) & 7) >> 2)) & 1) * 64)) + (((((((int)tidx) & 63) >> 5) + ((((int)tidx) & 3) >> 1)) & 1) * 32)) + (((((((int)tidx) & 31) >> 4) + (((int)tidx) & 1)) & 1) * 16)) + 64512), V+((((((int64_t)topk_indices[(((((((((int64_t)k) * (int64_t)128) + (((((int64_t)((int)blockIdx.z)) * ((int64_t)max_topk_num)) * ((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5)) * (int64_t)8)) + (((int64_t)i_3) * (int64_t)8)) + (((int64_t)((int)tidx)) >> (int64_t)4)) + ((((int64_t)head_to_reuse) * ((int64_t)max_topk_num)) * ((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5))) + (((((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5) - ((int64_t)((int)blockIdx.x))) - (int64_t)1) * ((int64_t)max_topk_num))) - (int64_t)8)]) * (int64_t)1024) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)tidx)) & (int64_t)15) * (int64_t)8)));
          }
        } else {
            if (tl::tl_shuffle_elect<128>()) {
              mbarrier[((k & 1) + 2)].expect_transaction(32768);
              tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 32768)])), 0, ((int)blockIdx.y), (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) - 32), ((int)blockIdx.z));
              tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 40960)])), 64, ((int)blockIdx.y), (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) - 32), ((int)blockIdx.z));
            }
        }
      }
      tl::mbarrier_cp_async_arrive(mbarrier[((k & 1) + 2)]);
      mbarrier[((k & 1) + 2)].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      *(float2*)(acc_o + (i_5 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      logsum[i_6] = 0x0p+0f/*0.000000e+00*/;
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_max[i_7] = -CUDART_INF_F;
    }
    tl::fence_proxy_async();
    mbarrier[8].wait(0);
    int condval_12;
    if ((((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) <= (topk_num + 32))) {
      condval_12 = ((((((seq_len + 31) >> 5) * 32) + 127) - (((int)blockIdx.x) * 32)) >> 7);
    } else {
      condval_12 = (((topk_num + 127) >> 7) + 1);
    }
    for (int k_1 = 0; k_1 < condval_12; ++k_1) {
      
      if (((((seq_len + 31) >> 5) * 32) - (((int)blockIdx.x) * 32)) <= (topk_num + 32)) {
        #pragma unroll
        for (int i_8 = 0; i_8 < 64; ++i_8) {
          float condval_16;
          if (((((((k_1 * 128) + (((i_8 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_8 & 1)) + 32) <= (((((((seq_len + 31) >> 5) * 32) + ((((int)threadIdx.x) >> 5) * 4)) + ((i_8 >> 5) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) - (((int)blockIdx.x) * 32)))) {
            condval_16 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_16 = -CUDART_INF_F;
          }
          acc_s[(((((i_8 & 31) >> 1) * 4) + ((i_8 >> 5) * 2)) + (i_8 & 1))] = condval_16;
        }
      } else {
        if (((k_1 * 128) + 128) <= topk_num) {
          #pragma unroll
          for (int i_9 = 0; i_9 < 32; ++i_9) {
            *(float2*)(acc_s + (i_9 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
          }
        } else {
            #pragma unroll
            for (int i_11 = 0; i_11 < 64; ++i_11) {
              float condval_21;
              if (((((((i_11 & 31) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_11 & 1)) <= ((((((int)threadIdx.x) >> 5) * 4) + ((i_11 >> 5) * 2)) + ((((int)threadIdx.x) & 31) >> 4)))) {
                condval_21 = 0x0p+0f/*0.000000e+00*/;
              } else {
                condval_21 = -CUDART_INF_F;
              }
              acc_s[(((((i_11 & 31) >> 1) * 4) + ((i_11 >> 5) * 2)) + (i_11 & 1))] = condval_21;
            }
        }
      }
      tl::fence_proxy_async();
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      tl::gemm_ss<128, 128, 128, 8, 1, 0, 1, 0, 128, 128, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[65536])), (&(((half_t*)buf_dyn_shmem)[((k_1 & 1) * 16384)])), (&(acc_s[0])));
      mbarrier[((k_1 & 1) + 4)].arrive();
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_max_prev[i_12] = scores_max[i_12];
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        scores_max[i_13] = -CUDART_INF_F;
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        #pragma unroll
        for (int rv = 0; rv < 32; ++rv) {
          scores_max[i_14] = max(scores_max[i_14], acc_s[((((rv & 15) * 4) + (i_14 * 2)) + (rv >> 4))]);
        }
        scores_max[i_14] = tl::AllReduce<tl::MaxOp, 4, 1, 0, 256>::run_hopper(scores_max[i_14]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_scale[i_15] = exp2f(((scores_max_prev[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 64; ++i_16) {
        acc_s[(((((i_16 & 31) >> 1) * 4) + ((i_16 >> 5) * 2)) + (i_16 & 1))] = exp2f(((acc_s[(((((i_16 & 31) >> 1) * 4) + ((i_16 >> 5) * 2)) + (i_16 & 1))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_16 >> 5)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_sum[i_17] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
          scores_sum[i_17] = (scores_sum[i_17] + acc_s[((((rv_1 & 15) * 4) + (i_17 * 2)) + (rv_1 >> 4))]);
        }
        scores_sum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0, 256>::run_hopper(scores_sum[i_17]);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        logsum[i_18] = ((logsum[i_18] * scores_scale[i_18]) + scores_sum[i_18]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 64; ++i_19) {
        acc_s_cast[(((((i_19 & 31) >> 1) * 4) + ((i_19 >> 5) * 2)) + (i_19 & 1))] = ((half_t)acc_s[(((((i_19 & 31) >> 1) * 4) + ((i_19 >> 5) * 2)) + (i_19 & 1))]);
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 64; ++i_20) {
        acc_o[(((((i_20 & 31) >> 1) * 4) + ((i_20 >> 5) * 2)) + (i_20 & 1))] = (acc_o[(((((i_20 & 31) >> 1) * 4) + ((i_20 >> 5) * 2)) + (i_20 & 1))] * scores_scale[(i_20 >> 5)]);
      }
      tl::fence_proxy_async();
      mbarrier[((k_1 & 1) + 2)].wait(((k_1 & 3) >> 1));
      tl::gemm_rs<128, 128, 128, 8, 1, 0, 0, 0, 128, 128, 0, 0, true>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 32768)])), (&(acc_o[0])));
      mbarrier[((k_1 & 1) + 6)].arrive();
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 64; ++i_21) {
      acc_o[(((((i_21 & 31) >> 1) * 4) + ((i_21 >> 5) * 2)) + (i_21 & 1))] = (acc_o[(((((i_21 & 31) >> 1) * 4) + ((i_21 >> 5) * 2)) + (i_21 & 1))] / logsum[(i_21 >> 5)]);
    }
    tl::__sync_thread_partial<3, 256>();
    #pragma unroll
    for (int i_22 = 0; i_22 < 8; ++i_22) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_22 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) + 65536)])), __pack_half2(((half_t)acc_o[(i_22 * 8)]), ((half_t)acc_o[((i_22 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_22 * 8) + 2)]), ((half_t)acc_o[((i_22 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_22 * 8) + 4)]), ((half_t)acc_o[((i_22 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_22 * 8) + 6)]), ((half_t)acc_o[((i_22 * 8) + 7)])));
    }
    tl::__sync_thread_partial<3, 256>();
    #pragma unroll
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      if (((((((seq_len + 31) >> 5) * 32) + (i_23 * 4)) + (((int)threadIdx.x) >> 6)) - (((int)blockIdx.x) * 32)) < (seq_len + 32)) {
        *(uint4*)(Output + ((((((((((((int64_t)seq_len) + (int64_t)31) >> (int64_t)5) * (int64_t)131072) + (((int64_t)i_23) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)4096)) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)) - (((int64_t)((int)blockIdx.x)) * (int64_t)131072)) - (int64_t)131072)) = *(uint4*)(((half_t*)buf_dyn_shmem) + (((i_23 * 2048) + (((int)threadIdx.x) * 8)) + 65536));
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
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 163840, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, int* __restrict__ topk_indices, int* __restrict__ head_mapping, float topk_percent, half_t* __restrict__ Output, int batch, int seq_len, int max_topk_num, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap K_desc;
	CUtensorMapDataType K_desc_type= (CUtensorMapDataType)6;
	cuuint32_t K_desc_tensorRank= 4;
	void *K_desc_globalAddress= K;
	cuuint64_t K_desc_globalDim[4]= {128,8,seq_len,batch};
	cuuint64_t K_desc_globalStride[4]= {2,256,2048,(int64_t)seq_len * 2048};
	cuuint32_t K_desc_boxDim[4]= {64,1,128,1};
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

	CUtensorMap V_desc;
	CUtensorMapDataType V_desc_type= (CUtensorMapDataType)6;
	cuuint32_t V_desc_tensorRank= 4;
	void *V_desc_globalAddress= V;
	cuuint64_t V_desc_globalDim[4]= {128,8,seq_len,batch};
	cuuint64_t V_desc_globalStride[4]= {2,256,2048,(int64_t)seq_len * 2048};
	cuuint32_t V_desc_boxDim[4]= {64,1,128,1};
	cuuint32_t V_desc_elementStrides[4]= {1,1,1,1};
	CUtensorMapInterleave V_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle V_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion V_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill V_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult V_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &V_desc, V_desc_type, V_desc_tensorRank, V_desc_globalAddress, V_desc_globalDim, V_desc_globalStride + 1, V_desc_boxDim, V_desc_elementStrides, V_desc_interleave, V_desc_swizzle, V_desc_l2Promotion, V_desc_oobFill);

	if (V_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor V_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	main_kernel<<<dim3((seq_len + 31) / 32, 8, batch), dim3(384, 1, 1), 163840, stream>>>(K, K_desc, Output, Q, V, V_desc, head_mapping, topk_indices, batch, max_topk_num, seq_len, topk_percent);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}

