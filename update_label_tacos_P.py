import h5py
from scipy.ndimage import zoom
import numpy as np
import os
from utils_weak import load_json, load_pickle, save_json, miou_two_dataset
from utils_weak import infer_idx, calculate_iou, preprocess, idx_time, time_idx, matadd
import sys
import shutil
import torch
from matplotlib import pyplot as plt
GUASSIAN_WIDTH = {"charades":   { "30": 1.5, "20": 4.0, "10": 4.0, "5": None},
                  "tacos":      { "30": 1.5, "20": None, "10": None, "5": None},
                 }

def batch_iou(candidates, gt):
    '''
    candidates: (prop_num, 2)
    gt: (2, )
    '''
    candidates = torch.from_numpy(candidates)
    gt = torch.from_numpy(gt)
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    res = inter.clamp(min=0) / union
    return res.numpy()



def get_index_vsl(sprob, eprob, outertype):
    sprob = np.expand_dims(sprob, axis=1)
    eprob = np.expand_dims(eprob, axis=0)

    if outertype == "matmul":
        matrix = np.matmul(sprob, eprob)
    elif outertype == "matadd":
        matrix = matadd(sprob, eprob)
    
    matrix = np.triu(matrix, k=0)
    sidx = np.argmax(np.max(matrix, axis=1))
    eidx = np.argmax(np.max(matrix, axis=0))
    return sidx, eidx


def nms_matrix(matrix, topk=5, thresh=0.5):
    matrix_ = matrix.flatten()
    idxs = np.argsort(matrix_)[::-1]
    moments = np.stack([idxs//matrix.shape[0], idxs%matrix.shape[1]]).T
    scores = np.sort(matrix_)[::-1]
    suppressed = np.zeros_like(scores, dtype="bool")
    count = 0
    for i in range(len(suppressed) - 1):
        if suppressed[i]:
            continue
        mask = batch_iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
        count += 1
        if count == topk:
            break
    return moments[~suppressed]

def get_props_nms(sprob, eprob, topk):
    sprob, eprob
    sprob = np.expand_dims(sprob, axis=1)
    eprob = np.expand_dims(eprob, axis=0)
    matrix = np.matmul(sprob, eprob)
    matrix = np.triu(matrix, k=0)
    res = nms_matrix(matrix, topk=topk, thresh=0.7)
    return res


def get_candidate_label(sprob, eprob, sbmn, ebmn, vlen):
    # res = get_props_nms(sprob, eprob, 6)

    if sbmn is None:
        res = get_props_nms(sprob, eprob, 6)
    else:
        res = get_props_nms(sprob, eprob, 6)
    tmp = res[:6].tolist()
    rr = []
    for i in tmp:
        rr.append([max(0, i[0]), min(vlen, i[1])])
    return rr


def extend_time(se, duration, alpha):
    s, e = se
    tmp = e - s
    s = s - tmp*alpha
    s = round(max(0, s), 2)
    e = e + tmp*alpha
    e = round(min(duration, e), 2)
    return [s, e]

    
import math
def get_gaussian_weight(center, vlen, max_vlen, alpha):
    x = np.linspace(-1, 1, num=max_vlen,  dtype=np.float32)
    sig = vlen / max_vlen
    sig *= alpha
    u = (center / (max_vlen-1)) * 2 - 1
    weight = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    weight /= np.max(weight)
    weight[vlen:] = 0.0
    return weight


def get_weight_gaussian_extend(idx_seed, L, width):
    s, e = idx_seed
    e = e + 1
    weight = np.zeros(L)
    weight[s:e] = 1

    width = (e-s) / L * width
    left_weight  = get_gaussian_weight(s, L, L, width)
    right_weight  = get_gaussian_weight(e, L, L, width)
    weight[:s] = left_weight[:s]
    weight[e:] = right_weight[e:]

    # plt.plot(weight)
    # plt.savefig("./images/soft.png")
    return weight

def get_weight(idx_news, L):
    res = []
    for idx in idx_news:
        s, e = idx
        weight = np.zeros(L)
        weight[s:e+1] = 1
        res.append(weight)
    return res

from numpy import linalg as la

def cosine_distance(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
def euclidSimilar(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))
def cross_entropy(a, b):
    import torch
    from torch.nn import functional as F
    a = torch.from_numpy(a).unsqueeze(0)
    b = torch.from_numpy(b).unsqueeze(0)
    return F.l1_loss(a, b).item()

def soft_iou(a, b):
    inter = a * b
    union= a + b - (a*b)
    res = inter.sum() / union.sum()
    return res



def select_pseudo_label(weight_seed, weight_news, threshold):
    dists = []
    for new in weight_news:
        dist = soft_iou(weight_seed, new)
        dists.append(dist)

    if np.max(dists) >= threshold:
        return np.argmax(dists)
    else:
        return -1

def main(old_data, seed_data, VMR_data, BMN_scores, threshold, width_extend):
    new_data = []
    for sample, seed_sample, predict in zip(old_data, seed_data, VMR_data):
        vid, duration, se_old, sent = sample[:4]
        _, _, se_seed, _, = seed_sample
        vlen = predict['vlen']
        sprob, eprob = predict["prop_logits"]
        L = sprob.shape[0]
        idx_seed = time_idx(se_seed, duration, vlen)
        if vid in BMN_scores.keys():
            sbmn, ebmn = BMN_scores[str(vid)][:]
        else:
            sbmn, ebmn = None, None
            
        sprob, eprob, sbmn, ebmn = preprocess(sprob, eprob, sbmn, ebmn, vlen)

        idx_news = get_candidate_label(sprob, eprob, sbmn, ebmn, vlen)
        weight_seed = get_weight_gaussian_extend(idx_seed, L, width_extend)
        weight_news = get_weight(idx_news, L)
        candidate_order = select_pseudo_label(weight_seed, weight_news, threshold=threshold)
        if candidate_order == -1:
            se_new = se_old
        else: 
            idx_new = idx_news[candidate_order]
            se_new = idx_time(idx_new, duration, vlen)
        record = [vid, duration, se_new, sent]
        new_data.append(record)
    return new_data

def cp_testjson(gt_path, new_path):
    gt_test = os.path.join(os.path.split(gt_path)[0], "test.json")
    new_test = os.path.join(os.path.split(new_path)[0], "test.json")
    shutil.copy(gt_test, new_test)


if __name__ == "__main__":

    task, P, THRESOLD, I = sys.argv[1:5]
    I = int(I)

    gt_path = './data/{}_gt/train.json'.format(task)
    seed_path = './data/{}_P{}_RE{}/train.json'.format(task, P, 0)
    new_path = './data/{}_P{}_T{}_RE{}/train.json'.format(task, P, THRESOLD, I)

    if I == 1:
        old_path = './data/{}_P{}_RE{}/train.json'.format(task, P, I-1)
        logist_path = "./results/{}/P{}_RE{}.pkl".format(task, P, I-1)
    else:
        old_path = './data/{}_P{}_T{}_RE{}/train.json'.format(task, P, THRESOLD, I-1)
        logist_path = "./results/{}/P{}_T{}_RE{}.pkl".format(task, P, THRESOLD, I-1)

    os.makedirs(os.path.split(new_path)[0], exist_ok=True)
    old_data = load_json(old_path)
    seed_data = load_json(seed_path)
    VMR_data = load_pickle(logist_path)
    BMN_scores = h5py.File("./results/bmn_{}.h5".format(task), 'r')

    THRESOLD =  int(THRESOLD)/100
    width_extend = GUASSIAN_WIDTH[task][P]
    new_data = main(old_data, seed_data, VMR_data, BMN_scores, THRESOLD, width_extend)

    save_json(new_data, new_path)
    cp_testjson(gt_path, new_path)
    old_miou, new_miou = miou_two_dataset(gt_path, old_path), miou_two_dataset(gt_path, new_path)
    print("{} -> {}".format(old_path, new_path))
    print("{:.4f} -> {:.4f}".format(old_miou, new_miou))
