import scipy.stats
import numpy as np
file_paths = {
    'safer':'/home/zjiehang/SparseNLP/result_log/agnews_bert/safer_logits.txt',
    'mask0.9':'/home/zjiehang/SparseNLP/result_log/agnews_roberta/mask_0.9_logit.txt',
    'mask0.7':'/home/zjiehang/SparseNLP/result_log/agnews_roberta/mask_0.7_logit.txt',
    'mask0.5':'/home/zjiehang/SparseNLP/result_log/agnews_roberta/mask_0.5_logit.txt',
    'mask0.3':'/home/zjiehang/SparseNLP/result_log/agnews_roberta/mask_0.3_logit.txt',
    'mask0.1':'/home/zjiehang/SparseNLP/result_log/agnews_roberta/mask_0.1_logit.txt',
    'mask0.05':'/home/zjiehang/SparseNLP/result_log/agnews_roberta/mask_0.05_logit.txt'
}

for key, value in file_paths.items():
    all_list = []
    with open(value, 'r', encoding='utf8') as file:
        for line in file.readlines():
            values = line.strip().split()
            all_list.append(scipy.stats.entropy([float(value) for value in values]))
    print('{}:{}'.format(key, np.mean(all_list)))
    
