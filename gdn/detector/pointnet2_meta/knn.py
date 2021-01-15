import numpy as np
import torch
import faiss

def search_index_pytorch_fast(index, x, k, nprobe=1, max_samples=5000):
    n, d = x.shape

    index.train(x[:max_samples])
    assert index.is_trained
    index.nprobe = nprobe
    index.add(x)
    D, I = index.search(x, k)
    index.reset()

    return D, I

'''
if __name__ == '__main__':
    import sys
    import time
    from sklearn.neighbors import NearestNeighbors

    d = 128

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, 8, faiss.METRIC_L2)

    while True:
        A = torch.randn(2048, d).cuda()

        cpu_start = time.time()
        A_cpu = A.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(A_cpu)
        D_cpu, I_cpu = nbrs.kneighbors(A_cpu)
        cpu_end = time.time()

        faiss_start = time.time()
        A_cpu = A.cpu().numpy()
        D, I = search_index_pytorch_fast(index, A_cpu, 5, nprobe=8)
        faiss_end = time.time()

        print('IVF accuracy = %.4f'%np.mean(I == np.array(I_cpu)))
        print('sklearn runtime: %.6f second'%(cpu_end-cpu_start))
        print('faiss runtime: %.6f second'%(faiss_end-faiss_start))
'''
