"""
This logic is largely taken from the examples in tile-ai/tilelang and modified
"""
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from tilelang.engine.param import KernelParam 
import argparse
import itertools
from tvm import tir, ir
from ..kernel_utils import *

torch.random.manual_seed(0)


def get_configs():
    block_N = [64, 128]
    block_H = [64]
    num_split = [2, 4, 8]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{
        'block_N': c[0],
        'block_H': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def flashattn(batch, heads, groups, seqlen_kv, dim, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dim]
    shape_o = [batch, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_N, block_H, num_split, num_stages, threads):
        part_shape = [batch, heads, num_split, dim]
        kv_group_num = 4
        valid_block_H = min(block_H, kv_group_num)

        @T.macro
        def flash_attn(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
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
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    # for i, j in T.Parallel(block_H, block_N):
                    #     acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                    #                                  -T.infinity(accum_dtype))
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
                    T.copy(V[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

        @T.macro
        def flash_attn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
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

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                total_chunks = T.ceildiv(seqlen_kv, block_N)
                base_chunks_per_split = T.floordiv(total_chunks, num_split)
                remainder_chunks = T.floormod(total_chunks, num_split)
                final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
                prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
                start_idx = prev_split_chunks * block_N
                for k in T.Pipelined(final_chunks, num_stages=num_stages):
                    T.copy(
                        K[bid, start_idx +
                          k * block_N:start_idx + (k + 1) * block_N,
                          cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(start_idx + k * block_N + j < seqlen_kv, acc_s[i, j],
                                                     -T.infinity(accum_dtype))
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
                    T.copy(
                        V[bid, start_idx +
                          k * block_N:start_idx + (k + 1) * block_N,
                          cur_kv_head, :], V_shared)
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
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, 128], dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    # lse_local: (local_id, thread_id)
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
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def flashattn_gqa_decode_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn(Q, K, V, Output)

        if num_split > 1:
            return flashattn_gqa_decode_split
        else:
            return flashattn_gqa_decode_no_split

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @jit(
            out_idx=[5],
            supply_type=tilelang.TensorSupplyType.Auto,
            ref_prog=ref_program,
            max_mismatched_ratio=0.05)
        def kernel(block_N=None, block_H=None, num_split=None, num_stages=None, threads=None):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

        return kernel()
    else:

        def kernel(block_N, block_H, num_split, num_stages, threads):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

        return kernel


def ref_program(query, key, value):
    """
    Inputs:
    - query (Tensor): [batch, heads, dim]
    - key (Tensor): [batch, seqlen_kv, groups, dim]
    - value (Tensor): [batch, seqlen_kv, groups, dim]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    from einops import rearrange, einsum
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    key = rearrange(key, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]
    value = rearrange(value, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

    query = rearrange(
        query, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, value,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out

def ref_program_fa(query, key, value):
    from flash_attn_interface import flash_attn_with_kvcache
    Q = query.unsqueeze(1)  # [batch_size, 1, heads, dim]
    out = flash_attn_with_kvcache(Q, key, value, causal=False)
    out = out.squeeze(1)  # [batch_size, heads, dim]
    return out

def ref_program_correct(query, key, value):
    """
    Inputs:
    - query (Tensor): [batch, heads, dim]
    - key (Tensor): [batch, seqlen_kv, groups, dim]
    - value (Tensor): [batch, seqlen_kv, groups, dim]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    q = query.unsqueeze(2)  # [batch_size, heads, 1, dim]
    k = key.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen_kv, dim]
    v = value.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen_kv, dim]

    attn_weights = torch.matmul(q, k.transpose(2, 3)) / scale  # [batch_size, heads, 1, seqlen_kv]
    
    attn_weights = torch.softmax(attn_weights, dim=-1)  # [batch_size, heads, 1, seqlen_kv]
    out = torch.matmul(attn_weights, v)  # [batch_size, heads, 1, dim]
    out = out.squeeze(2)  # [batch_size, heads, dim]
    return out


def main(batch: int = 1,
         heads: int = 32,
         groups: int = 8,
         kv_seqlen: int = 8192,
         dim: int = 128,
         tune: bool = False,
         run_type: str = "benchmark"  # "benchmark", "correctness"
         ):
    qk_flops = 2 * batch * heads * kv_seqlen * dim
    pv_flops = 2 * batch * heads * kv_seqlen * dim
    total_flops = qk_flops + pv_flops

    if (not tune):

        def get_heuristic_config() -> dict:
            # Get CUDA device properties
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = torch.cuda.current_device()
            sm_major, sm_minor = torch.cuda.get_device_capability(device)
            sm_version = sm_major * 10 + sm_minor
            # print(f"CUDA device capability: {sm_version}")
            if sm_version == 89:
                return {
                    "block_N": 128,
                    "block_H": 64,
                    "num_split": 8,
                    "num_stages": 0,
                    "threads": 128
                }
            else:
                return {
                    "block_N": 128,
                    "block_H": 64,
                    "num_split": 8,
                    "num_stages": 2,
                    "threads": 128
                }

        config = get_heuristic_config()
        program = flashattn(T.symbolic("batch"), T.symbolic("heads"), T.symbolic("groups"), T.symbolic("kv_seqlen"), dim, tune=tune)(**config)
        kernel = tilelang.compile(program, out_idx=[5])
        # cuda_source = kernel.get_kernel_source()  
        # print(cuda_source)

        do_benchm = (run_type == "benchmark")
        do_corr = (run_type == "correctness")

        if do_corr:
            num_runs = 100
        else:
            num_runs = 10

        lat_f_sum = lat_og_sum = 0.0
        cnt_f = cnt_og = 0

        for _ in range(num_runs):
            if do_corr:
                kv_seqlen_sample = torch.randint(32, kv_seqlen+1, (1,)).item()
            else:
                kv_seqlen_sample = kv_seqlen
            q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
            k = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
            v = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
            glse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float16)
            Output_partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float16)

            if do_corr:
                o_ref = ref_program(q, k, v)
                o_ref_correct = ref_program_correct(q, k, v)
                o_og = kernel(q, k, v, glse, Output_partial)
                assert_similar(o_ref, o_og, name="ref_og", assert_=True, print_=False)
                assert_similar(o_og, o_ref_correct, name="o_ref_correct_og", assert_=True, print_=False)

            if do_benchm:
                try:
                    latency_f = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v])
                    lat_f_sum += latency_f
                    cnt_f += 1
                except Exception:
                    pass
                try:
                    latency_og = do_bench(kernel, n_warmup=1, n_repeat=5, input_tensors=[q, k, v, glse, Output_partial])
                    lat_og_sum += latency_og
                    cnt_og += 1
                except Exception:
                    pass
            del q, k, v, glse, Output_partial
            torch.cuda.empty_cache()

        if do_bench:
            avg_f = lat_f_sum / cnt_f if cnt_f > 0 else 0.0
            avg_og = lat_og_sum / cnt_og if cnt_og > 0 else 0.0

            results = [f"{kv_seqlen}"]
            tflops_f = total_flops / avg_f * 1e-9 if avg_f > 0 else 0
            tflops_og = total_flops / avg_og * 1e-9 if avg_og > 0 else 0
            results.extend([f"{avg_f:.2f}", f"{tflops_f:.2f}"])
            results.extend([f"{avg_og:.2f}", f"{tflops_og:.2f}"])
            print(",".join(results))
    else:
        best_result = flashattn(batch, heads, groups, kv_seqlen, dim, tune=tune)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--kv_seqlen', type=int, default=8192, help='kv sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    parser.add_argument('--run_type', type=str, default="benchmark", choices=["benchmark", "correctness"], help='Type of run: benchmark, correctness')
    args = parser.parse_args()
    main(args.batch, args.heads, args.groups, args.kv_seqlen, args.dim, args.tune, args.run_type)
