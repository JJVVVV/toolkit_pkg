import numpy as np

from . import MetricDict


def mrr(all_logits):
    all_logits = np.array(all_logits)  # (b, 5)
    # argsort返回一个数组`a`, `a[i]`表示第i大的元素在原列表中的index,
    # 也就是如果要将原数组从小到大排序, 则第i个位置应该放arr[a[i]]
    # 换句话说, 原数组中, 第i个元素排序后所在的位置, 就是a中值等于i的元素的下标,
    # 例如: a[index] = i, 表示原数组中第i个元素, 排序后排在第index位
    sorted_indices = np.argsort(-all_logits, axis=1)
    # nozero 返回非零元素的位置, 例如对于二维数组, 会返回一个元组(rows: Array, cols: Array), 其中rows代表行号, cols代表列号,
    # (rows[i], cols[i])指示了一个非零元素对应的下标位置
    position_pos = np.nonzero(sorted_indices == 0)[1] + 1
    mrr = (1 / position_pos).mean().item()
    return MetricDict(MRR=mrr)


def hit_at_1(all_logits):
    all_logits = np.array(all_logits)  # (b, 5)
    # argsort返回一个数组`a`, `a[i]`表示第i大的元素在原列表中的index,
    # 也就是如果要将原数组从小到大排序, 则第i个位置应该放arr[a[i]]
    # 换句话说, 原数组中, 第i个元素排序后所在的位置, 就是a中值等于i的元素的下标,
    # 例如: a[index] = i, 表示原数组中第i个元素, 排序后排在第index位
    sorted_indices = np.argsort(-all_logits, axis=1)
    # nozero 返回非零元素的位置, 例如对于二维数组, 会返回一个元组(rows: Array, cols: Array), 其中rows代表行号, cols代表列号,
    # (rows[i], cols[i])指示了一个非零元素对应的下标位置
    position_pos = np.nonzero(sorted_indices == 0)[1] + 1
    hit_1 = (position_pos == 1).mean().item()
    return MetricDict({"Hit@1": hit_1})
