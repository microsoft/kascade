# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import os  
import hashlib  
import shutil  
from pathlib import Path  
import tilelang  
from tilelang.jit.adapter.libgen import LibraryGenerator  
from tilelang.utils.target import determine_target  
from typing import Callable, List, Literal, Optional, Union
from functools import partial
  
def compile_custom_kernel_from_cu(  
    func,  
    cu_file_path,  
    so_dir,  
    out_idx=None,
    execution_backend="cython",  
    pass_configs=None, 
    verbose=False
):  
    """  
    Takes a .cu file, checks if compilation is needed, compiles if needed,  
    moves to custom path, and creates a JITKernel for execution.  
    """  
    # Read the CUDA source file  
    with open(cu_file_path, "r") as f:  
        cuda_source = f.read()  
      
    # Generate hash of the source code for change detection  
    source_hash = hashlib.sha256(cuda_source.encode()).hexdigest()  
      
    # Create paths for caching  
    cache_dir = os.path.dirname(so_dir)  
    os.makedirs(cache_dir, exist_ok=True)  
    filepath = cu_file_path.split("/")[-1].replace(".cu", "")
    hash_file_path = os.path.join(cache_dir, f"{os.path.basename(filepath)}.hash") 
    so_file_path = os.path.join(so_dir, f"{os.path.basename(filepath)}.so") 

    # Check if compilation is needed  
    need_compile = True  
    if os.path.exists(so_file_path) and os.path.exists(hash_file_path):  
        with open(hash_file_path, "r") as f:  
            cached_hash = f.read().strip()  
            if cached_hash == source_hash:  
                need_compile = False   

    kernel_og = tilelang.compile(func, out_idx=None, pass_configs=pass_configs)
      
    if need_compile:  
        # Create library generator and compile  
        target_obj = determine_target(kernel_og.target)  
        lib_gen = LibraryGenerator(target_obj, verbose=verbose)  
        lib_gen.assign_pass_configs(pass_configs)
        lib_gen.update_lib_code(cuda_source)  
          
        # Compile to temporary location first  
        lib_gen.compile_lib()  
        temp_so_path = lib_gen.get_lib_path()  
          
        # Move to custom location  
        shutil.move(temp_so_path, so_file_path)  
          
        # Save hash for future checks  
        with open(hash_file_path, "w") as f:  
            f.write(source_hash)    
      
    # Create and return JITKernel  
    return tilelang.JITKernel.from_database(  
        func=func,  
        kernel_global_source=cuda_source,  
        kernel_lib_path=so_file_path,  
        params=kernel_og.params,  
        target=kernel_og.target,  
        target_host=kernel_og.target_host,  
        out_idx=out_idx,  
        execution_backend=execution_backend,  
        pass_configs=kernel_og.pass_configs,  
    )  

def print_red_warning(msg):
    print(f"\033[91m{msg}\033[0m")


def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f'{name} all zero')
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_similar(x, y, eps=1e-2, name="tensor", assert_=False, print_=True):
    sim = calc_sim(x, y, name)
    diff = 1. - sim
    if not (0 <= diff <= eps):
        print_red_warning(f'{name} Error: {diff}')
        if assert_:
            raise AssertionError(f'{name} Error: {diff}')
    else:
        if print_:
            print(f'passed: {name} diff={diff}')
            print((x-y).abs().max(), (x-y).abs().mean(), (x-y).abs().std())

def assert_equal(x, y, name="tensor", assert_=False, print_=True):
    if not torch.equal(x, y):
        mismatches = (x != y).sum().item()
        ttl_elems = x.numel()
        if (mismatches / ttl_elems) * 100 > 15:
            print_red_warning(f'{name} Error: {mismatches} mismatches out of {ttl_elems}')
            #print position of first mismatch
            # Find first mismatch index for each row in dim=-2
            mismatches = (x != y)
            # Reshape to (batch * heads, seqlen) if needed
            if mismatches.dim() == 3:
                batch, heads, seqlen = mismatches.shape
                mismatches_2d = mismatches.view(-1, seqlen)
            else:
                mismatches_2d = mismatches
            
            # Find first mismatch for each row
            first_mismatches = []
            for i in range(mismatches_2d.shape[0]):
                row_mismatches = mismatches_2d[i].nonzero(as_tuple=False)
                if row_mismatches.numel() > 0:
                    first_mismatches.append((i, row_mismatches[0].item()))
            
            if first_mismatches:
                # Find the row with lowest mismatch index
                min_row, min_idx = min(first_mismatches, key=lambda x: x[1])
                if mismatches.dim() == 3:
                    batch_idx = min_row // heads
                    head_idx = min_row % heads
                    print_red_warning(f'{name} first mismatch at batch={batch_idx}, head={head_idx}, seq_idx={min_idx}')
                else:
                    print_red_warning(f'{name} first mismatch at row={min_row}, idx={min_idx}')
            if assert_:
                raise AssertionError(f'{name} Error: {mismatches} mismatches out of {ttl_elems}')
        elif print_:
            print(f'passed: {name} with {mismatches/ttl_elems * 100}% mismatches')
    else:
        if print_:
            print(f'passed: {name} equal')

def assert_allclose(x, y, eps=1e-2, name="tensor", assert_=False, print_=True):
    all_close = torch.allclose(x, y, atol=eps, rtol=eps)
    print(name + "  all_close={}".format(all_close))
    if not all_close:
        diff = (x - y).abs()
        print("all_close={}, max={}, min={}, mean={}".format(all_close, diff.max().item(), diff.min().item(), diff.mean().item()))
        max_indices = torch.nonzero(diff == diff.max().item())
        first_index = tuple(max_indices[0].tolist())
        print(f"Index: {first_index}, expect: {x[first_index]}, actual: {y[first_index]}")
        print("failed test!")
        exit(1)

def make_tile_causal_mask(batch: int,
                          seq_len: int,
                          tile_size: int,
                          *,
                          dtype=torch.float16,
                          device="cuda") -> torch.Tensor:
    """
    Returns a mask of shape [batch, 1, num_tiles, seq_len] filled with
    0 (keep) or -inf (mask-out) exactly like the original code, but
    without first materialising a (tile_size×) larger tensor.
    """
    num_tiles = (seq_len + tile_size - 1) // tile_size           # T
    col_ids   = torch.arange(seq_len,  device=device)            # [S]
    row_ids   = (torch.arange(num_tiles, device=device)          # [T]
                 * tile_size + (tile_size - 1))                  # last row in each tile

    # [T, S]  True where j > i  (upper-triangular part to be masked)
    upper = row_ids[:, None] < col_ids

    # convert to desired dtype / value
    mask = torch.where(upper,
                       torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
                       torch.zeros((), dtype=dtype, device=device))

    # add batch-/head dims   (no real memory, just views)
    mask = mask.unsqueeze(0).unsqueeze(0)        # [1,1,T,S]
    mask = mask.expand(batch, 1, num_tiles, seq_len)

    # roll & zero-out first tile as in the original code
    mask = torch.roll(mask, shifts=1, dims=2)
    mask[:, :, 0, :] = torch.finfo(dtype).min
    return mask

def sample_topk_indices_tilewise(batch: int,
                                 groups: int,
                                 seq_len: int,
                                 tile_size: int,
                                 topk: int,
                                 *,
                                 device: str = "cuda",
                                 dtype: torch.dtype = torch.int32
                                 ) -> torch.Tensor:
    """
    Exact replacement for

        scores  = torch.rand(B,G,T,S)
        scores += make_tile_causal_mask(...)
        topk_ix = torch.topk(scores, k=topk, dim=-1).indices

    but the peak memory is only
        B · G · topk            (for a single tile)  +  output,
    independent of `seq_len`.

    Output
        Tensor [batch, groups, num_tiles, topk] (int32)
    """
    num_tiles = (seq_len + tile_size - 1) // tile_size
    out = torch.empty(batch, groups, num_tiles, topk,
                      dtype=dtype, device=device)

    # ------------------------------------------------------------
    # tile 0 : all scores == -inf  →  deterministic 0..topk-1
    # ------------------------------------------------------------
    first = torch.arange(topk, device=device, dtype=dtype)       # [topk]
    out[:, :, 0, :] = first.view(1, 1, -1).expand(batch, groups, -1)

    # ------------------------------------------------------------
    # tiles 1 … num_tiles-1
    # ------------------------------------------------------------
    for t in range(1, num_tiles):
        hi = min(seq_len, t * tile_size)     # highest VALID column (exclusive)

        if topk <= hi:
            # need only topk unique valid indices  →  pick random without replacement
            sel = torch.randperm(hi, device=device, dtype=dtype)[:topk]
        else:
            # all valid positions must appear; fill the rest with the first
            # invalid columns, exactly what torch.topk would output when the
            # remaining scores are -inf
            pad = torch.arange(hi, hi + (topk - hi), device=device, dtype=dtype)
            sel = torch.cat((torch.randperm(hi, device=device, dtype=dtype), pad))[:topk]

        out[:, :, t, :] = sel.view(1, 1, -1).expand(batch, groups, -1)

    return out

def check_for_nans(o, o_ref):
    nan_mask = torch.isnan(o)
    if nan_mask.any():
        first_idx = nan_mask.nonzero()[0]
        b, s = first_idx[0].item(), first_idx[1].item()
        print(f"Found NaN in o at batch={b}, seq_index={s}")
        print("o row:", o[b, s])
        print("o_ref row:", o_ref[b, s])
    else:
        print("No NaNs found in o")

def softmax_(x: torch.Tensor, dim: int = -1, div: bool = True) -> torch.Tensor:
    """
    In-place numerically-stable softmax along dimension `dim`.
    Modifies `x` and also returns it for convenience.
    """
    # 1. subtract max
    # note: .values if using torch.max; keepdim=True to allow broadcasting
    # max_vals = x.max(dim=dim, keepdim=True).values
    max_vals = x.max()
    x.sub_(max_vals)

    # 2. exponentiate in-place
    x.exp_()

    # 3. divide by sum of exps
    if div:
        sum_vals = x.sum(dim=dim, keepdim=True)
        x.div_(sum_vals)

    return x


def do_bench(
    func: Callable,
    input_tensors: List[torch.Tensor],
    n_warmup: int = 1,
    n_repeat: int = 1,
    grad_to_none: Optional[List[torch.Tensor]] = None,
    quantiles: Optional[List[float]] = None,
    fast_flush: bool = True,
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
) -> Union[float, List[float]]:
    """Benchmarks the runtime of a PyTorch function.

    This function handles:
    - L2 cache flushing between runs for consistent timing
    - Automatic warmup and repeat count calculation
    - Optional gradient clearing for backward passes
    - Multiple measurement modes (mean, median, min, max)

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds
        rep: Target number of repetitions
        n_warmup: Override for number of warmup iterations
        n_repeat: Override for number of timing iterations
        grad_to_none: Tensors whose gradients should be cleared between runs
        quantiles: Optional performance percentiles to compute
        fast_flush: Whether to use faster L2 cache flushing
        return_mode: How to aggregate timing results ("mean", "median", "min", "max")

    Returns:
        float: Aggregated runtime in milliseconds
    """
    assert return_mode in ["min", "max", "mean", "median"]
    fn = partial(func, *input_tensors)
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")


    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()

def to_gb_str(bytes):
    return f"{bytes / (1024 * 1024 * 1024)} GB"

def print_gpu_mem_stats(loc):
    _, total_mem = torch.cuda.mem_get_info()
    allocated_mem = torch.cuda.memory_allocated()
    reserved_mem = torch.cuda.memory_reserved()
    free_mem = total_mem - allocated_mem
    print(f"GPU mem stats at {loc}: Total: {to_gb_str(total_mem)}, Reserved: {to_gb_str(reserved_mem)}, Allocated: {to_gb_str(allocated_mem)}, Free: {to_gb_str(free_mem)}")