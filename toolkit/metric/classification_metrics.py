import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def praf(preds, labels):
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    # print(type(cm))
    print(cm)
    tn, fp, fn, tp = cm.ravel()

    # 计算精确率
    precision = tp / (tp + fp)
    # 计算召回率
    recall = tp / (tp + fn)
    # 计算准确率
    accuracy = np.trace(cm) / cm.sum()
    # 计算f1
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return precision, recall, accuracy, f1
