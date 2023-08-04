import torch
import torch.nn.functional as F

# def compute_kl_loss(p, q, pad_mask=None, temperature=1.0):
#     """计算kl散度损失"""
#     p = p / temperature
#     q = q / temperature
#     loss1 = F.kl_div(F.log_softmax(p, dim=-1), F.log_softmax(q, dim=-1), reduction="batchmean", log_target=True)
#     loss2 = F.kl_div(F.log_softmax(q, dim=-1), F.log_softmax(p, dim=-1), reduction="batchmean", log_target=True)
#     return (loss1 + loss2) / 2


def kl_loss(p: torch.Tensor, q: torch.Tensor, temperature=1.0) -> torch.Tensor:
    """计算kl散度损失"""
    p = p / temperature
    q = q / temperature
    loss1 = F.kl_div(F.log_softmax(p, dim=-1), F.log_softmax(q, dim=-1), reduction="batchmean", log_target=True)
    loss2 = F.kl_div(F.log_softmax(q, dim=-1), F.log_softmax(p, dim=-1), reduction="batchmean", log_target=True)
    return (loss1 + loss2) / 2


def cos_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算cos损失"""
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 1 - cos(p, q)
    return loss.sum()


def mse_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算mse损失"""
    loss = F.mse_loss(p, q, reduction="none")
    loss = loss.sum(-1)
    return loss.mean()
