import sys
from glob import glob
import json
import pandas as pd
import numpy as np

def find_num_pt(s):
    filename_split = s.split('_')
    for idx, ss in enumerate(filename_split):
        if ss.lower().endswith('pt'):
            break
    return int(filename_split[idx][:-2])

if __name__ == '__main__':
    prefix = sys.argv[1]
    eval_files = glob(prefix+'/**/eval.json')
    outputs = {'mAP@5': [], 'mAP@10': [], 'mAP@15': [], 'npts': []}
    row_names = []
    for path in eval_files:
        dirname = path.split('/')[-2]
        num_pt = find_num_pt(dirname)
        outputs['npts'].append(num_pt)
        with open(path, 'rb') as fp:
            eval_data = json.load(fp)
        for threshold in list(eval_data['overall']):
            outputs['mAP@%s'%threshold].append(eval_data['overall'][threshold])
        for obj, ap3 in eval_data['object'].items():
            for threshold, value in ap3.items():
                col_name = '%s@%s'%(obj, threshold)
                if not col_name in outputs:
                    outputs[col_name] = []
                outputs[col_name].append(value)
        row_names.append(dirname)
    output_csv = pd.DataFrame()
    for key, value in outputs.items():
        output_csv[key] = value
    output_csv.index = row_names
    output_csv.sort_values(by=['npts', 'mAP@5'], kind='mergesort', inplace=True, ascending=[True, False])
    print(output_csv.head())
    output_csv.to_csv('summarized_vs_points.csv')
