# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse
from ..kernel_utils import *

torch.random.manual_seed(0)

def flashattn(heads, groups, dim):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // groups

    def kernel_func(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen, topk):
        shape_q = [batch, heads, dim]
        shape_k = [batch, max_cache_seqlen, groups, dim]
        shape_v = [batch, max_cache_seqlen, groups, dim]
        shape_o = [batch, heads, dim]
        shape_topk = [batch, groups, topk]
        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)

        @T.macro
        def flash_attn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlens: T.Tensor([batch], "int32"),
                topk_indices: T.Tensor(shape_topk, "int32"),
                head_mapping: T.Tensor([groups], "int32"),
                topk_percent: T.float32,
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)
                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                head_to_reuse = head_mapping[cur_kv_head]
                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                topk_num = T.ceildiv(T.int32((topk_percent / 100) * (cache_seqlens[bid])), block_N)*128
                total_chunks = T.ceildiv(topk_num, block_N)
                base_chunks_per_split = T.floordiv(total_chunks, num_split)
                remainder_chunks = T.floormod(total_chunks, num_split)
                final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
                prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
                start_idx = prev_split_chunks * block_N
                for k in T.Pipelined(final_chunks, num_stages=num_stages):
                    for i, j in T.Parallel(block_N, dim):
                        idx = topk_indices[bid, head_to_reuse, start_idx + k * block_N + i]
                        with T.attr(T.int32(0), "async_scope", 1): 
                            K_shared[i, j] = K[bid, idx, cur_kv_head, j]
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i,j in T.Parallel(block_N, dim):
                        idx = topk_indices[bid, head_to_reuse, start_idx + k * block_N + i]
                        with T.attr(T.int32(0), "async_scope", 1): 
                            V_shared[i, j] = V[bid, idx, cur_kv_head, j]

                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                
                if final_chunks > 0:
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(logsum[:valid_block_H],
                       glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], accum_dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, 128], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    lse_local:
                        T.Fragment(lse_local.shape, forward_fn=lambda i, j: (j, i)),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                for k, j in T.Parallel(num_split, 128):
                    lse_local[k, j] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def flashattn_gqa_decode_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlens: T.Tensor([batch], "int32"),
                topk_indices: T.Tensor(shape_topk, "int32"),
                head_mapping: T.Tensor([groups], "int32"),
                topk_percent: T.float32,
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, cache_seqlens, topk_indices, head_mapping, topk_percent, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return flashattn_gqa_decode_split


    def kernel(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen, topk):
        return kernel_func(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen, topk)

    return kernel

def ref_program_fa(query, key, value):
    from flash_attn_interface import flash_attn_with_kvcache
    Q = query.unsqueeze(1)  # [batch_size, 1, heads, dim]
    out = flash_attn_with_kvcache(Q, key, value, causal=True)
    out = out.squeeze(1)  # [batch_size, heads, dim]
    return out

def ref_program_correct(query, key, value, cache_seqlens, topk_indices, head_mapping, topk_percent):
    """
    Inputs:
    - query (Tensor): [batch, heads, dim]
    - key (Tensor): [batch, seqlen_kv, groups, dim]
    - value (Tensor): [batch, seqlen_kv, groups, dim]
    - mask (Tensor): [batch, seqlen_kv, groups]
    - topk_indices (Tensor): [batch, groups, topk]
    - head_mapping (Tensor): [groups]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5

    q = query.unsqueeze(2)  # [batch_size, heads, 1, dim]
    topk_num = ((((topk_percent * cache_seqlens)/100).to(torch.int32) + 127) // 128) * 128

    #select after attention
    indices = topk_indices[:, head_mapping, :].unsqueeze(2)  # [batch, groups, 1, topk]
    for b in range(indices.shape[0]):
        indices[b, :, 0, topk_num[b]:] = indices[b, :, 0, topk_num[b]-1].unsqueeze(-1).expand_as(indices[b, :, 0, topk_num[b]:])
    indices = indices.repeat_interleave(num_head_groups, dim=1).to(torch.int64)  # [batch, heads, 1, topk]
    k = key.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, kv_seqlen, dim]
    v = value.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, kv_seqlen, dim]


    attn_weights = torch.matmul(q, k.transpose(2, 3)) / scale  # [batch_size, heads, 1, seqlen_kv]

    # select after attention
    values = attn_weights.gather(3, indices)  # [batch_size, heads, 1, topk]
    attn_weights.fill_(float('-inf'))  # Reset all attention weights
    attn_weights.scatter_(3, indices, values)  # Set the topk attention weights
    
    attn_weights = torch.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)  # [batch_size, heads, 1, seqlen_kv]
    out = torch.matmul(attn_weights, v)  # [batch_size, heads, 1, dim]
    out = out.squeeze(2)  # [batch_size, heads, dim]
        
    return out

def main(batch: int = 1,
         heads: int = 32,
         groups: int = 8,
         max_cache_seqlen: int = 8192,
         dim: int = 128,
         topk: int = 10,
         run_type: str = "benchmark",
         code: bool = False,
         with_ref: bool = False):

    program = flashattn(heads, groups, dim)(
        batch=T.symbolic("batch"),
        block_N=128,
        block_H=64,
        num_split=8,
        num_stages=2,
        threads=128,
        max_cache_seqlen=T.symbolic("max_cache_seqlen"),
        topk=T.symbolic("topk"),
    )
    kernel = tilelang.compile(program, out_idx=[9], pass_configs={ "tl.disable_safe_memory_legalize": True , })
    if code:
        cuda_source = kernel.get_kernel_source()  
        print(cuda_source)
        return
    # Create a new kernel using your compiled library  
    modified_kernel = compile_custom_kernel_from_cu(
        func=program,  
        cu_file_path="./kascade_decode_cu_kernels/dynamic_alllen_reuse_cp_async_dim128_gsize4_heads32_groups8_with_kvcache.cu",
        so_dir="./kascade_decode_cu_kernels/__socache__/",
        out_idx=[9],  
        execution_backend="cython",  
        pass_configs={ "tl.disable_safe_memory_legalize": True , }, 
    )
    
    if run_type == "correctness":
        runs = 100
    else:
        runs = 10

    # For benchmark averaging
    sum_ref_e_latency = 0.0
    sum_ref_f_latency = 0.0
    sum_tile_og_latency = 0.0
    sum_tile_async_latency = 0.0

    cnt_ref_e = 0
    cnt_ref_f = 0
    cnt_tile_og = 0
    cnt_tile_async = 0

    for _ in range(runs):
        if run_type == "correctness":
            cache_seqlens = torch.randint(128, max_cache_seqlen, (batch,), device='cuda', dtype=torch.int32)
            topk_sample = torch.randint(10, 50, size=(1,)).item()
        else:
            cache_seqlens = torch.full((batch,), max_cache_seqlen, device='cuda', dtype=torch.int32)
            topk_sample = topk
        max_seqlen = cache_seqlens.max().item()
        max_topk = (int((topk_sample * max_seqlen)/100) + 127) // 128 * 128
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, max_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, max_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        head_mapping = torch.randint(0, groups, (groups,), device="cuda", dtype=torch.int32)
        random_scores = torch.rand(batch, groups, max_seqlen, device="cuda")
        topk_indices = torch.topk(random_scores, k=max_topk, dim=-1).indices.to(torch.int32)
        del random_scores
        glse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)

        if run_type == "correctness":
            print(f"cache_seqlens={cache_seqlens}, topk={topk_sample}")
            o = modified_kernel(q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample, glse, Output_partial)
            glse.zero_()
            Output_partial.zero_()
            o_og = kernel(q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample, glse, Output_partial)
            o_ref = ref_program_correct(q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample)
            eps = 1e-3
            assert_ = False
            print_ = True
            assert_similar(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps)
            assert_allclose(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps)
            assert_similar(o, o_og, name="o_og", assert_=assert_, print_=print_, eps=eps)
            assert_allclose(o, o_og, name="o_og", assert_=assert_, print_=print_, eps=eps)
            assert_similar(o_og, o_ref, name="og_ref", assert_=assert_, print_=print_, eps=eps)
            assert_allclose(o_og, o_ref, name="og_ref", assert_=assert_, print_=print_, eps=eps)
        else:
            if with_ref:
                try:
                    ref_e_latency = do_bench(
                    ref_program_correct, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample]
                    )
                    sum_ref_e_latency += ref_e_latency
                    cnt_ref_e += 1
                except Exception:
                    pass

                try:
                    ref_f_latency = do_bench(
                    ref_program_fa, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v]
                    )
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

                try:
                    tile_og_latency = do_bench(
                    kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample, glse, Output_partial]
                    )
                    sum_tile_og_latency += tile_og_latency
                    cnt_tile_og += 1
                except Exception:
                    pass

            try:
                tile_async_latency = do_bench(modified_kernel,
                n_warmup=1, n_repeat=5,
                input_tensors=[q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample, glse, Output_partial]
                )
                sum_tile_async_latency += tile_async_latency
                cnt_tile_async += 1
            except Exception:
                pass
        del q, k, v, head_mapping, topk_indices, glse, Output_partial
        torch.cuda.empty_cache()

    if run_type == "benchmark":
        qk_flops = 2 * batch * heads * max_cache_seqlen * dim
        pv_flops = 2 * batch * heads * max_cache_seqlen * dim
        total_flops = qk_flops + pv_flops

        topk_qk_flops = 2 * batch * heads * max_topk * dim
        topk_pv_flops = 2 * batch * heads * max_topk * dim
        total_topk_flops = topk_qk_flops + topk_pv_flops
        if with_ref:
            avg_ref_e_latency = (sum_ref_e_latency / cnt_ref_e) if cnt_ref_e else 0.0
            avg_ref_e_tflops = (total_topk_flops / avg_ref_e_latency * 1e-9) if avg_ref_e_latency > 0 else 0.0
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            avg_ref_f_tflops = (total_flops / avg_ref_f_latency * 1e-9) if avg_ref_f_latency > 0 else 0.0
            avg_tile_og_latency = (sum_tile_og_latency / cnt_tile_og) if cnt_tile_og else 0.0
            avg_tile_og_tflops = (total_topk_flops / avg_tile_og_latency * 1e-9) if avg_tile_og_latency > 0 else 0.0
        avg_tile_async_latency = (sum_tile_async_latency / cnt_tile_async) if cnt_tile_async else 0.0
        avg_tile_async_tflops = (total_topk_flops / avg_tile_async_latency * 1e-9) if avg_tile_async_latency > 0 else 0.0

        if with_ref:
            print(f"{max_cache_seqlen},{topk},"
            f"{avg_ref_e_latency:.2f},{avg_ref_e_tflops:.2f},"
            f"{avg_ref_f_latency:.2f},{avg_ref_f_tflops:.2f},"
            f"{avg_tile_og_latency:.2f},{avg_tile_og_tflops:.2f},"
            f"{avg_tile_async_latency:.2f},{avg_tile_async_tflops:.2f}")
        else:
            print(f"{max_cache_seqlen},{topk},"
            f"{avg_tile_async_latency:.2f},{avg_tile_async_tflops:.2f}")
    else:
        print(f"{runs} Correctness test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--max_cache_seqlen', type=int, default=8192, help='kv sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--topk', type=float, default=10, help='topk percent for attention')
    parser.add_argument('--run_type', type=str, default='benchmark', choices=["benchmark", "correctness"], help='Type of run: benchmark, correctness')
    parser.add_argument('--code', action='store_true', help='output code configs')
    parser.add_argument('--with_ref', action='store_true', help='run with reference for benchmark')
    args = parser.parse_args()
    main(args.batch, args.heads, args.groups, args.max_cache_seqlen, args.dim, args.topk, args.run_type, args.code, args.with_ref)