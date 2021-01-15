import numpy as np
import torch
import faiss

def swigPtrFromTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch trensor (on CPU or GPU) """
    assert x.is_contiguous()

    if x.dtype == torch.float32:
        return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() + x.storage_offset() * 4)

    if x.dtype == torch.int64:
        return faiss.cast_integer_to_idx_t_ptr(x.storage().data_ptr() + x.storage_offset() * 8)

    raise Exception("tensor type not supported: {}".format(x.dtype))

def search_index_pytorch(x, k):
    """call the search function of an index with pytorch tensor I/O (CPU
        and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()

    D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    I = torch.empty((n, k), dtype=torch.int64, device=x.device)

    res = faiss.StandardGpuResources()
    res.noTempMemory()
    gpu_index = faiss.GpuIndexFlatL2(res, d)

    torch.cuda.synchronize()
    xptr = swigPtrFromTensor(x)
    Iptr = swigPtrFromTensor(I)
    Dptr = swigPtrFromTensor(D)
    gpu_index.add_c(n, xptr)
    gpu_index.search_c(n, xptr, k, Dptr, Iptr)
    torch.cuda.synchronize()

    return D, I

def search_index_pytorch_fast(x, k, nlist=1024, nprobe=256, max_samples=5000):
    """call the search function of an index with pytorch tensor I/O (CPU
        and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()

    D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    I = torch.empty((n, k), dtype=torch.int64, device=x.device)

    quantizer = faiss.IndexFlatIP(d)
    res = faiss.StandardGpuResources()
    res.noTempMemory()
    gpu_index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)

    torch.cuda.synchronize()
    xptr = swigPtrFromTensor(x)
    Iptr = swigPtrFromTensor(I)
    Dptr = swigPtrFromTensor(D)
    gpu_index.train_c(min(max_samples, n), xptr)
    assert gpu_index.is_trained
    gpu_index.nprobe = nprobe
    gpu_index.add_c(n, xptr)
    gpu_index.search_c(n, xptr, k, Dptr, Iptr)
    torch.cuda.synchronize()

    return D, I

if __name__ == '__main__':
    import sys
    import time
    from sklearn.neighbors import NearestNeighbors
    A = torch.randn(4096, 128).cuda()
    A_cpu = A.cpu().numpy()
    cpu_start = time.time()
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(A_cpu)
    D_cpu, I_cpu = nbrs.kneighbors(A_cpu)
    cpu_end = time.time()

    gpu_start = time.time()
    D, I = search_index_pytorch(A, 5)
    gpu_end = time.time()

    ivf_start = time.time()
    D_ivf, I_ivf = search_index_pytorch_fast(A, 5, nlist=1, nprobe=8)
    ivf_end = time.time()

    print('Integrity check? ', np.all(I.cpu().numpy() == np.array(I_cpu)))
    print('IVF accuracy = %.4f'%np.mean(I_ivf.cpu().numpy() == np.array(I_cpu)))
    print('cpu runtime: %.6f second'%(cpu_end-cpu_start))
    print('gpu runtime: %.6f second'%(gpu_end-gpu_start))
    print('gpu(fast) runtime: %.6f second'%(ivf_end-ivf_start))
