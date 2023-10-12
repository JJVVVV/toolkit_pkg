import math
import random

import numpy as np


def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def map(y_true, y_pred, rel_threshold=0):
    # 比如q1对应相关的d排名是1，2，5，7（假设q1有4个相关d），那么对于q1的ap（average
    # precision）的计算就是（1 / 1 + 2 / 2 + 3 / 5 + 4 / 7）
    s = 0
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):  # j是编号，（g,p)是y_true, y_pred
        if g > rel_threshold:  # g>0就是正类
            ipos += 1.0  # pos + 1 相当于TP+1
            s += ipos / (j + 1.0)  # 等式右边是precision? s相当于AP

    if ipos == 0:
        s = 0.0
    else:
        s /= ipos  # 求MAP
    return s


def mrr(y_true, y_pred, rel_threshold=0.0):
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    s = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            s = 1 / (j + 1.0)
            break
    return s


def ndcg(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.0):
        if k <= 0.0:
            return 0.0
        s = 0.0
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c_g = sorted(c, key=lambda x: x[0], reverse=True)
        c_p = sorted(c, key=lambda x: x[1], reverse=True)
        idcg = 0.0
        ndcg = 0.0
        for i, (g, p) in enumerate(c_g):
            if i >= k:
                break
            if g > rel_threshold:
                idcg += (math.pow(2.0, g) - 1.0) / math.log(2.0 + i)
        for i, (g, p) in enumerate(c_p):
            if i >= k:
                break
            if g > rel_threshold:
                ndcg += (math.pow(2.0, g) - 1.0) / math.log(2.0 + i)
        if idcg == 0.0:
            return 0.0
        else:
            return ndcg / idcg

    return top_k


def precision(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.0):
        if k <= 0:
            return 0.0
        s = 0.0
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        prec = 0.0
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                prec += 1
        prec /= k
        return prec

    return top_k


# compute recall@k
# the input is all documents under a single query
def recall(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.0):
        if k <= 0:
            return 0.0
        s = 0.0
        y_true = _to_list(np.squeeze(y_true).tolist())  # y_true: the ground truth scores for documents under a query
        y_pred = _to_list(np.squeeze(y_pred).tolist())  # y_pred: the predicted scores for documents under a query
        pos_count = sum(i > rel_threshold for i in y_true)  # total number of positive documents under this query
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        recall = 0.0
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                recall += 1
        recall /= pos_count
        return recall

    return top_k


def mse(y_true, y_pred, rel_threshold=0.0):
    s = 0.0
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    return np.mean(np.square(y_pred - y_true), axis=-1)


def acc(y_true, y_pred):
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    assert y_true_idx.shape == y_pred_idx.shape
    return 1.0 * np.sum(y_true_idx == y_pred_idx) / len(y_true)


import glob
import heapq
import os
from collections import defaultdict

from tensorboard.backend.event_processing import event_accumulator

# seeds_dir = "runs/roberta-base/QQP-Describe/question_with_describe/0k-2k/baseline_dev_0-200_with_describe_prompt3/5/16/2e-05/"


def load_metrics_from_tb(seeds_dir):
    metric_dict = defaultdict(list)
    seed_dirs = glob.glob(seeds_dir + "/*")
    success = 0
    for seed_dir in seed_dirs:
        # print(seed_dir)
        try:
            event_dir = glob.glob(os.path.join(seed_dir, "logs", "hparam*"))[0]
        except:
            try:
                event_dir = glob.glob(os.path.join(seed_dir, "logs", "1*"))[0]
            except Exception as e:
                print(seed_dir)
                continue

        # print(event_dir)
        assert os.path.exists(event_dir), f"event_dir: {event_dir} not exists"
        event_acc = event_accumulator.EventAccumulator(event_dir)
        event_acc.Reload()
        # print(event_acc.scalars.Keys())

        # 获取标量指标
        scalar_tags = event_acc.Tags()["scalars"]
        for tag in scalar_tags:
            scalar_events = event_acc.Scalars(tag)
            for event in scalar_events:
                # event.step
                metric_dict[tag].append(event.value)
        success += 1
    print(f"total: {len(seed_dirs)}\nsuccess: {success}")
    return metric_dict


def calculate_mean_with_metric_dict(metric_dict):
    return {key: sum(value) / len(value) for key, value in metric_dict.items()}


# TODO 有点bug, 例如当设置top_k=5时, 会分别计算top5的acc与top5的f1, 而top5的acc所对应的种子与top5的f1所对应的种子可能不同
def calculate_top_k_mean_with_metric_dict(metric_dict, top_k=5):
    for key in metric_dict:
        if metric_dict[key]:
            metric_dict[key] = heapq.nlargest(top_k, metric_dict[key])
    return {key: sum(value) / len(value) for key, value in metric_dict.items()}
