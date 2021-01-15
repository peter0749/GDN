import numpy as np
import torch
import faiss

#def search_index_pytorch_fast(gpu_index, x, k, nprobe=1, max_samples=5000):
def search_index_pytorch_fast(gpu_index, x, k):
    n, d = x.shape

    gpu_index.reset() # clear the graph
    #gpu_index.train(x[:max_samples])
    #assert gpu_index.is_trained
    #gpu_index.nprobe = nprobe
    gpu_index.add(x)
    D, I = gpu_index.search(x, k)

    return D, I

if __name__ == '__main__':
    import sys
    import time
    from sklearn.neighbors import NearestNeighbors

    d = 128

    #cfg = faiss.GpuIndexIVFFlatConfig()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True
    res = faiss.StandardGpuResources()
    res.setTempMemory(64 * 1024 * 1024)
    #gpu_index = faiss.GpuIndexIVFFlat(res, d, 1, faiss.METRIC_L2, cfg)
    gpu_index = faiss.GpuIndexFlatL2(res, d, cfg)

    A = torch.randn(6144, d).cuda()

    cpu_start = time.time()
    A_cpu = A.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(A_cpu)
    D_cpu, I_cpu = nbrs.kneighbors(A_cpu)
    cpu_end = time.time()

    gpu_start = time.time()
    A_cpu = A.cpu().numpy()
    D, I = search_index_pytorch_fast(gpu_index, A_cpu, 5)
    gpu_end = time.time()

    print('IVF accuracy = %.4f'%np.mean(I == np.array(I_cpu)))
    print('cpu runtime: %.6f second'%(cpu_end-cpu_start))
    print('gpu runtime: %.6f second'%(gpu_end-gpu_start))
