import os
import sys
import glob
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
#from sklearn.decomposition import PCA

if __name__=='__main__':
    input_file = sys.argv[1]
    if not os.path.exists('visualization_embedding'):
        os.makedirs('visualization_embedding')
    plot_info = []
    method_set = {}
    method_list = []
    with open(input_file, 'r') as fp:
        for line in fp:
            s = line.split()
            plot_info.append((s[0], s[1], int(s[2]), int(s[3]))) # method name, path to features, max_num_sample, is_GT
            method_set[s[0]] = (s[4], s[5]) # color, marker
            method_list.append(s[0])
    file_list = list(map(lambda x: x.split('/')[-2:], glob.glob(plot_info[0][1]+'/**/*features.npy', recursive=True)))
    obj_set = {}
    for obj_name, feat_id in file_list:
        if not obj_name in obj_set:
            obj_set[obj_name] = []
        obj_set[obj_name].append(feat_id)
    for obj_name, feat_ids in obj_set.items():
        target_labels = []
        all_features = None
        for info in plot_info:
            method, feature_path, max_num_sample, is_gt = info
            for feat_id in feat_ids:
                feature = np.load(feature_path+'/'+obj_name+'/'+feat_id)
                feature = feature[~np.all(feature==0, axis=1)]
                if len(feature)>max_num_sample:
                    if is_gt==1:
                        feature = feature[np.random.choice(len(feature), max_num_sample, replace=False)]
                    else:
                        feature = feature[:max_num_sample] # preserve order
                target_labels += [method]*len(feature)
                if all_features is None:
                    all_features = feature
                else:
                    all_features = np.append(all_features, feature, axis=0)
        target_labels = np.asarray(target_labels)
        embeddings = TSNE(n_jobs=11, perplexity=20, n_iter=2000, verbose=1).fit_transform(all_features)
        #embeddings = PCA(n_components=2).fit_transform(all_features)
        for method in method_list:
            color = method_set[method][0][1:-1].split(',') #(255,255,255)
            color = np.array([int(color[0]), int(color[1]), int(color[2])], dtype=np.float32) / 255.0
            marker = method_set[method][1]
            ind = (target_labels==method)
            print('%s : %d'%(method, np.sum(ind)))
            vis = embeddings[ind]
            plt.scatter(vis[:,0], vis[:,1], color=color, alpha=1.0, marker=marker, label=method)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.title(obj_name)
        plt.savefig('visualization_embedding/%s.png'%obj_name)
        #plt.show()
        plt.clf()
