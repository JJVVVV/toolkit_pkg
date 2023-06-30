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
