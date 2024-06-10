import numpy as np

from . import MetricDict


def mrr(all_scores, gold_index=0):
    """
    all_scores, (batch_size, num_scores): 每行一个列表包括num_scores个分数
    gold_index: 指出num_scores个分数中, 哪个(下标)应该排第一
    """
    all_scores = np.array(all_scores)  # (b, 5) [[0.8, 0.9, 0.1, 0.4, 0.3], ...]
    # argsort返回一个数组`a`, `a[i]`表示第i小(大)的元素在原数组中的index, 排序后数组[i] = 原数组[a[i]].
    # 例如: a[i] = index, 表示原数组中下标为index的元素, 排序后排在第i位. 也即, 原数组中下标为index的元素是整个数组中第i小(大)的元素.
    sorted_indices = np.argsort(-all_scores, axis=1)  # 从大到小排序index, [[1, 0, 3, 4, 2], ...]
    # nozero 返回非零元素的位置, 例如对于二维数组, 会返回一个元组(rows: Array, cols: Array), 其中rows代表行号, cols代表列号,
    # (rows[i], cols[i])指示了一个非零元素对应的下标位置  (rows[0], cols[0]) = (0, 1)
    position_pos = np.nonzero(sorted_indices == gold_index)[1] + 1  # [2, ...]
    mrr = (1 / position_pos).mean().item()
    return MetricDict(MRR=mrr)


def hit_at_1(all_scores, gold_index=0):
    """
    all_scores, (batch_size, num_scores): 每行一个列表包括num_scores个分数
    gold_index: 指出num_scores个分数中, 哪个(下标)应该排第一
    """
    all_scores = np.array(all_scores)  # (b, 5) [[0.8, 0.9, 0.1, 0.4, 0.3], ...]
    # argsort返回一个数组`a`, `a[i]`表示第i小(大)的元素在原数组中的index, 排序后数组[i] = 原数组[a[i]].
    # 例如: a[i] = index, 表示原数组中下标为index的元素, 排序后排在第i位. 也即, 原数组中下标为index的元素是整个数组中第i小(大)的元素.
    sorted_indices = np.argsort(-all_scores, axis=1)  # 从大到小排序index, [[1, 0, 3, 4, 2], ...]
    # nozero 返回非零元素的位置, 例如对于二维数组, 会返回一个元组(rows: Array, cols: Array), 其中rows代表行号, cols代表列号,
    # (rows[i], cols[i])指示了一个非零元素对应的下标位置  (rows[0], cols[0]) = (0, 1)
    position_pos = np.nonzero(sorted_indices == gold_index)[1] + 1  # [2, ...]
    hit_1 = (position_pos == 1).mean().item()
    return MetricDict({"Hit@1": hit_1})


# def calculate_metric_callback(all_labels, all_logits, mean_loss):
#     # all_labels = np.array(all_labels)  # ()
#     all_logits = np.array(all_logits)  # (b, 5)
#     sorted_indices = np.argsort(-all_logits, axis=1)
#     position_pos = np.nonzero(sorted_indices == 0)[1] + 1
#     hit_1 = (position_pos == 1).mean() * 100
#     mrr = (1 / position_pos).mean() * 100
#     return MetricDict({"Hit@1": hit_1, "MRR": mrr, "Loss": mean_loss})
