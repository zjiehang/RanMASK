import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
probs_path = "/home/zjiehang/SparseNLP/result_log/agnews_roberta/statistics-sparse-len128-epo10-batch32-rate{}-probs.txt"
mask_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9]
colors_list = ['r', 'g', 'b']
shape_list = ['o-', 's-', '^-']


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

def read_from_file(file_path: str):
    return_dict = {}
    with open(file_path, 'r', encoding="utf8") as file:
        for line in file.readlines():
            line = line.strip()
            key, value = line.split(":  ")
            value = list(map(float, value.split()))
            return_dict[key] = value
    return return_dict

for rate, color, shape in zip(mask_rate_list, colors_list, shape_list):
    x = np.arange(1, 11)
    dicts = read_from_file(probs_path.format(rate))
    
    p_i_x = np.array(dicts["pix"])
    js_values = []
    for i in range(1, 11):
        js = JS_divergence(p_i_x, np.array(dicts[str(i)]))
        js_values.append(js)

    y = np.array(js_values)
    plt.plot(x, y, shape, color=color, label="Retained {:.0f}%".format((1.0 - rate) * 100))

plt.xlabel("Perturbed Numbers")
plt.ylabel("JS divergence")
plt.legend(loc="best")
plt.show()
