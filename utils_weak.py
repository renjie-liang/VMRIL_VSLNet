import json
import pickle
import h5py
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import numpy as np
import os


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def load_json(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
    # with open(filename, mode='r') as f:
        data = json.load(f)
    return data

def save_json(data, filename):
    with open(filename, mode='w', encoding='utf-8') as f:
        json.dump(data, f)


def load_lines(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        return [e.strip("\n") for e in f.readlines()]

def load_pickle(filename):
    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)
        return data

def time_idx(t, duration, vlen):
    if isinstance(t, list):
        res = []
        for i in t:
            res.append(time_idx(i, duration, vlen))
        return res
    else:
        return round(t / duration * (vlen - 1))

def idx_time(t, duration, vlen):
    if isinstance(t, list) or isinstance(t, tuple):
        res = []
        for i in t:
            res.append(idx_time(i, duration, vlen))
        return res
    else:
        return round(t / (vlen-1) * duration, 2)

def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    if (union[1] - union[0]) == 0.0:
        return 0.0
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def infer_idx(start_prob, end_prob):
    start_prob = torch.from_numpy(start_prob)
    end_prob = torch.from_numpy(end_prob)
    outer = torch.matmul(start_prob.unsqueeze(1),end_prob.unsqueeze(0))
    outer = torch.triu(outer, diagonal=0)
    _, new_s_dix = torch.max(torch.max(outer, dim=1)[0], dim=0)  # (batch_size, )
    _, new_e_dix = torch.max(torch.max(outer, dim=0)[0], dim=0)  # (batch_size, )
    return new_s_dix.item(), new_e_dix.item()


def matadd(x, y):
    x = x.repeat(y.shape[1], axis=1)
    y = y.repeat(x.shape[0], axis=0)
    return x + y



def preprocess(sprob_in, eprob_in, sbmn_in, ebmn_in, vlen):
    sprob, eprob = sigmoid(sprob_in), sigmoid(eprob_in)
    sprob[vlen:], eprob[vlen:] = 0, 0 

    if sbmn_in is None:
        sbmn, ebmn = None, None
    else:
        sbmn = np.zeros_like(sprob)
        sbmn[:vlen] = zoom(sbmn_in, vlen/len(sbmn_in))
        ebmn = np.zeros_like(eprob)
        ebmn[:vlen] = zoom(ebmn_in, vlen/len(ebmn_in))
    
    return sprob, eprob, sbmn, ebmn



def miou_two_dataset(path1, path2):
    with open(path1, mode='r') as f:
        data1 = json.load(f)
    with open(path2, mode='r') as f:
        data2 = json.load(f)
    assert len(data1) == len(data2), "{} {}".format(len(data1), len(data2))

    miou = []
    for x1, x2 in zip(data1, data2):
        assert x1[0] == x2[0]
        iou = calculate_iou(x1[2], x2[2])
        miou.append(iou)
    return np.mean(miou)