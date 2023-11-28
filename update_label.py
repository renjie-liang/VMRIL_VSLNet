import h5py
from scipy.ndimage import zoom
import numpy as np
import os
from utils_weak import load_json, load_pickle, save_json, miou_two_dataset
from utils_weak import infer_idx, calculate_iou, preprocess, idx_time, matadd
import sys
import shutil
import torch

FACTOR = {
    "vmr": [1.0],
    "bmn": [1.0]}

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


def updata_label(sprob, eprob, sbmn, ebmn, vlen):
    # res = get_props_nms(sprob, eprob, 6)
    if sbmn is None:
        res = get_props_nms(sprob, eprob, 6)
    else:
        res = get_props_nms(sprob*sbmn, eprob*ebmn, 6)
    # return np.concatenate([res1[:3], res2[:3]]).tolist()
    tmp = res[:6].tolist()
    rr = []
    for i in tmp:
        rr.append([max(0, i[0]-1), min(vlen, i[1]+1)])
    return rr


def iou_select(time_news, se_seed, se_old, IOU):
    ious = [calculate_iou(i, se_seed) for i in time_news]

    if max(ious) > IOU:
        return time_news[np.argmax(ious)]
    else:
        return se_old

def iou_select_combine(time_news, se_seed, se_old, IOU):
    left_candidates, right_candidates = [], []

    for i in time_news:
        ll = i[:]
        ll[1] = min(ll[1], se_seed[1])
        left_candidates.append(ll)

        rr = i[:]
        rr[0] = max(rr[0], se_seed[0])
        right_candidates.append(rr)

    left_ious = [calculate_iou(i, se_seed) for i in left_candidates]
    right_ious = [calculate_iou(i, se_seed) for i in right_candidates]
    new_se = se_seed[:]
    if max(left_ious) > IOU:
        new_se[0] = left_candidates[np.argmax(left_ious)][0]
    if max(right_ious) > IOU:
        new_se[1] = right_candidates[np.argmax(right_ious)][1]
    return new_se

def extend_time(se, duration, alpha):
    s, e = se
    tmp = e - s
    s = s - tmp*alpha
    s = round(max(0, s), 2)
    e = e + tmp*alpha
    e = round(min(duration, e), 2)
    return [s, e]

def main(old_data, seed_data, VMR_data, BMN_scores, IOU):
    new_data = []
    for sample, seed_sample, predict in zip(old_data, seed_data, VMR_data):
        vid, duration, se_old, sent = sample[:4]
        _, _, se_seed, _, = seed_sample
        se_seed = extend_time(se_seed, duration, 1.5)

        vlen = predict['vlen']
        sprob, eprob = predict["prop_logits"]
        if vid in BMN_scores.keys():
            sbmn, ebmn = BMN_scores[str(vid)][:]
        else:
            sbmn, ebmn = None, None
            
        sprob, eprob, sbmn, ebmn = preprocess(sprob, eprob, sbmn, ebmn, vlen)

        idx_news = updata_label(sprob, eprob, sbmn, ebmn, vlen)
        time_news = idx_time(idx_news, duration, vlen)

        se_new = iou_select(time_news, se_seed, se_old, IOU)
        # print(se_seed, se_new)
        # print(time_news)
        # se_new = iou_select_combine(time_news, se_seed, se_old, IOU)
        # print(se_new)
        record = [vid, duration, se_new, sent]
        new_data.append(record)
    return new_data

def cp_testjson(gt_path, new_path):
    gt_test = os.path.join(os.path.split(gt_path)[0], "test.json")
    new_test = os.path.join(os.path.split(new_path)[0], "test.json")
    shutil.copy(gt_test, new_test)


if __name__ == "__main__":

    task = "charades"
    task, P, IOU, I = sys.argv[1:5]
    I = int(I)

    gt_path = './data/{}_gt/train.json'.format(task)
    seed_path = './data/{}_P{}_RE{}/train.json'.format(task, P, 0)
    new_path = './data/{}_P{}_IOU{}_RE{}/train.json'.format(task, P, IOU, I)

    if I == 1:
        old_path = './data/{}_P{}_RE{}/train.json'.format(task, P, I-1)
        logist_path = "./results/{}/P30_RE{}.pkl".format(task, I-1)
    else:
        old_path = './data/{}_P{}_IOU{}_RE{}/train.json'.format(task, P, IOU, I-1)
        logist_path = "./results/{}/P{}_IOU{}_RE{}.pkl".format(task, P, IOU, I-1)

    os.makedirs(os.path.split(new_path)[0], exist_ok=True)

    old_data = load_json(old_path)
    seed_data = load_json(seed_path)
    VMR_data = load_pickle(logist_path)
    BMN_scores = h5py.File("./results/bmn_charades.h5", 'r')

    IOU =  int(IOU)/100
    new_data = main(old_data, seed_data, VMR_data, BMN_scores, IOU)

    save_json(new_data, new_path)
    cp_testjson(gt_path, new_path)
    old_miou, new_miou = miou_two_dataset(gt_path, old_path), miou_two_dataset(gt_path, new_path)
    print("{} -> {}".format(old_path, new_path))
    print("{:.4f} -> {:.4f}".format(old_miou, new_miou))
