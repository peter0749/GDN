import os
import sys
import glob
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    input_file = sys.argv[1]
    if not os.path.exists('visualization_embedding'):
        os.makedirs('visualization_embedding')
    max_num_pred_per_pc = 10
    max_num_gt_per_pc   = 30
    plot_info = []
    method_set = {}
    method_list = []
    with open(input_file, 'r') as fp:
        for line in fp:
            s = line.split()
            if len(s)==3: # GT
                plot_info.append((s[0], s[1], None)) # method name, path to features, path to score
                method_set[s[0]] = s[2] # color
                method_list.append(s[0])
            elif len(s)==4: # pred
                plot_info.append((s[0], s[1], s[2]))
                method_set[s[0]] = s[3]
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
        scores = None
        for info in plot_info:
            method, feature_path, score_path = info
            local_scores = []
            for feat_id in feat_ids:
                if score_path is None: # Ground truth
                    feature = np.load(feature_path+'/'+obj_name+'/'+feat_id)
                    feature = feature[np.all(feature==0, axis=1)]
                    if len(feature)>max_num_gt_per_pc:
                        feature = feature[np.random.choice(len(feature), max_num_gt_per_pc, replace=False)]
                    local_scores += [0]*len(feature)
                    target_labels += [method]*len(feature)
                else:
                    feature = np.load(feature_path+'/'+obj_name+'/'+feat_id)
                    feature_ind = []
                    with open(score_path+'/'+obj_name+'/'+feat_id[:-13]+'.meta', 'rb') as fp:
                        meta = pickle.load(fp)
                        meta_scores = []
                        for n,t in enumerate(meta):
                            if n>=max_num_pred_per_pc:
                                break
                            score = t[0]
                            if score>-np.inf:
                                feature_ind.append(n)
                                meta_scores.append(score)
                                target_labels.append(method)
                    feature = feature[feature_ind]
                    local_scores += meta_scores
                if all_features is None:
                    all_features = feature
                else:
                    all_features = np.append(all_features, feature, axis=0)
            local_scores = np.asarray(local_scores, dtype=np.float32)
            if len(local_scores)>0:
                local_scores = (local_scores-local_scores.min()) / (local_scores.max()-local_scores.min()+1e-8)
            if scores is None:
                scores = local_scores
            else:
                scores = np.append(scores, local_scores, axis=0)
        target_labels = np.asarray(target_labels)
        if len(target_labels)>20000:
            _, target_labels, _, scores, _, all_features = train_test_split(target_labels, scores, all_features, test_size=20000, stratify=target_labels, shuffle=True)
        embeddings = TSNE(n_jobs=30, perplexity=20, n_iter=2000, verbose=1).fit_transform(all_features)
        for method in method_list:
            color = method_set[method][1:-1].split(',') #(255,255,255)
            color = np.array([int(color[0]), int(color[1]), int(color[2])], dtype=np.float32) / 255.0
            ind = (target_labels==method)
            print('%s : %d'%(method, len(ind)))
            vis = embeddings[ind]
            s   = np.clip(np.asarray(scores[ind])**0.3333, 0.5, 1.0)
            if method!='GT':
                plt.scatter(vis[0:1,0], vis[0:1,1], c=color, alpha=1.0, marker='.', label=method)
                plt.scatter(vis[:,0], vis[:,1], c=np.append(np.tile(color[np.newaxis], (len(s), 1)), s.reshape(-1, 1), axis=1), marker='.')
            else:
                plt.scatter(vis[:,0], vis[:,1], color=color, alpha=1.0, marker='.', label=method)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.title(obj_name)
        plt.savefig('visualization_embedding/%s.png'%obj_name)
        plt.clf()
        #plt.show()
