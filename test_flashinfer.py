import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["WORLD_SIZE"] = "4"

import torch
import flashinfer
import time
import random
from tqdm.auto import tqdm
import gc


num_qo_heads = 32
hidden_size = 4096
num_kv_heads = 8
head_dim = hidden_size // num_qo_heads
max_num_pages = 128
page_size = 16


def get_random_cfg(nnz_qo=1024, batch_size=2, dtype=torch.float16):

    qo_indptr = list(range(1, nnz_qo))
    random.shuffle(qo_indptr)
    qo_indptr = sorted(qo_indptr[:batch_size - 1])
    qo_indptr = torch.tensor([0] + qo_indptr + [nnz_qo], dtype=torch.int32, device="cuda:0") 
    # [0, 33, 44, 55, 66, 77, 88, nnz_qo]

    paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")

    paged_kv_indptr = list(range(1, max_num_pages))
    random.shuffle(paged_kv_indptr)
    paged_kv_indptr = sorted(paged_kv_indptr[:batch_size - 1])
    paged_kv_indptr = torch.tensor([0] + paged_kv_indptr + [max_num_pages], dtype=torch.int32, device="cuda:0")
    # [0, 17, 29, 44, 48, 66, 100, 128]

    # 1 <= paged_kv_last_page_len <= page_size
    paged_kv_last_page_len = torch.randint(1, page_size + 1, [batch_size], dtype=torch.int32, device="cuda:0")
    # [1, 7, 14, 4, 3, 1, 16]

    q = torch.randn(nnz_qo, num_qo_heads, head_dim, dtype=dtype).to("cuda:0")
    kv_cache = torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda:0"
    )

    return {
        "plan": {
            "qo_indptr": qo_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_last_page_len": paged_kv_last_page_len
        },
        "run": {
            "q": q,
            "paged_kv_cache": kv_cache
        }

    }


# def clone_cfg(cfg):
#     return {
#         "plan": {
#             "qo_indptr": cfg["plan"]["qo_indptr"].clone(),
#             "paged_kv_indices": cfg["plan"]["paged_kv_indices"].clone(),
#             "paged_kv_indptr": cfg["plan"]["paged_kv_indptr"].clone(),
#             "paged_kv_last_page_len": cfg["plan"]["paged_kv_last_page_len"].clone()
#         },
#         "run": {
#             "q": cfg["run"]["q"].clone(),
#             "paged_kv_cache": cfg["run"]["paged_kv_cache"].clone()
#         }
#     }


def run_wrapper(cfg, reduction=False, runs=10000):

    # allocate 128MB workspace buffer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    
    start = time.time()
    # create auxiliary data structures for batch prefill attention
    prefill_wrapper.plan(
        **cfg['plan'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        causal=True,
        allow_fp16_qk_reduction=reduction
    )
    
    # compute batch prefill attention, reuse auxiliary data structures
    for i in range(runs):
        o = prefill_wrapper.run(**cfg['run'])

    end = time.time()
    return end - start, o


def clean_cache():
    time.sleep(5)
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    q_len = 2048

    cfg = get_random_cfg(nnz_qo=q_len, dtype=torch.float16)

    clean_cache()
    print("########### float16 tensor w/o reduction")
    run_wrapper(cfg, reduction=False, runs=1)

    clean_cache()
    print("########### float16 tensor w/ reduction")
    run_wrapper(cfg, reduction=True, runs=1)

    clean_cache()
    cfg = get_random_cfg(nnz_qo=q_len, dtype=torch.bfloat16)

    clean_cache()
    print("########### bfloat16 tensor w/o reduction")
    run_wrapper(cfg, reduction=False, runs=1)

    clean_cache()
    print("########### bfloat16 tensor w/ reduction")
    run_wrapper(cfg, reduction=True, runs=1)


# ncu --set full --replay-mode kernel -f -o test_flashinfer python test_flashinfer.py --devices 0