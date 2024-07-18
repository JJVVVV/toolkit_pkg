import torch
import torch.nn.functional as F

# def compute_kl_loss(p, q, pad_mask=None, temperature=1.0):
#     """计算kl散度损失"""
#     p = p / temperature
#     q = q / temperature
#     loss1 = F.kl_div(F.log_softmax(p, dim=-1), F.log_softmax(q, dim=-1), reduction="batchmean", log_target=True)
#     loss2 = F.kl_div(F.log_softmax(q, dim=-1), F.log_softmax(p, dim=-1), reduction="batchmean", log_target=True)
#     return (loss1 + loss2) / 2


def kl_loss(p: torch.Tensor, q: torch.Tensor, temperature=1.0, reduction="batchmean") -> torch.Tensor:
    """计算kl散度损失"""
    p = p / temperature
    q = q / temperature
    loss1 = F.kl_div(F.log_softmax(p, dim=-1), F.log_softmax(q, dim=-1), reduction=reduction, log_target=True)
    loss2 = F.kl_div(F.log_softmax(q, dim=-1), F.log_softmax(p, dim=-1), reduction=reduction, log_target=True)
    return (loss1 + loss2) / 2


def cos_loss(p: torch.Tensor, q: torch.Tensor, reduction="mean") -> torch.Tensor:
    """计算cos损失"""
    loss = 1 - F.cosine_similarity(p, q, dim=-1)
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        return loss


def mse_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算mse损失"""
    loss = F.mse_loss(p, q, reduction="none")
    loss = loss.sum(-1)
    return loss.mean()


class ContrastLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05, margin: float = 0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin


# Implementation according to https://spaces.ac.cn/archives/8847
class CoSentLoss(ContrastLoss):
    "用于单塔模型"

    def __init__(self, temperature: float = 0.05, margin: float = 0):
        super().__init__(temperature, margin)

    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor, text_embeddings_neg: torch.Tensor) -> torch.Tensor:
        "输入为三元组(x, pos, neg), 分别对应3个参数"
        # text_embddings: (n, bedding_size)
        # text_pos_embeddings: (n, bedding_size)
        # text_neg_embeddings: (n, bedding_size)
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_embeddings_pos, dim=-1) / self.temperature  # (n)
        sim_neg_vector = torch.cosine_similarity(text_embeddings, text_embeddings_neg, dim=-1) / self.temperature  # (n)
        sim_matrix_diff = sim_neg_vector[:, None] - sim_pos_vector[None, :]
        dif = torch.cat([sim_matrix_diff.view(-1), torch.zeros(1, dtype=sim_matrix_diff.dtype, device=sim_matrix_diff.device)])
        loss = torch.logsumexp(dif, dim=0) / text_embeddings.size(0)

        return loss


class CoSentLoss_logits(ContrastLoss):
    "用于双塔模型"

    def __init__(self, temperature: float = 0.05, margin: float = 0):
        super().__init__(temperature, margin)

    def forward(self, pos_pair_logits: torch.Tensor, neg_pair_logits: torch.Tensor) -> torch.Tensor:
        """
        三元组推广版：(x, pos, neg) -> (x, num_pos*pos, num_neg*neg)
        batch_size个样本, 每个样本有num_pos个正例和num_neg个负例, 当num_pos=num_neg=1时, 退化为三元组
        pos_pair_logits: (batch_size, num_pos) batch_size个样本与其各自对应的num_pos个正例之间的logtis
        neg_pair_logits: (batch_size, num_neg) batch_size个样本与其各自对应的num_neg个负例之间的logtis
        """
        pos_pair_score = torch.sigmoid(pos_pair_logits) / self.temperature  # (b, num_pos)
        neg_pair_score = torch.sigmoid(neg_pair_logits) / self.temperature  # (b, num_neg)
        # (b, num_neg, 1, 1) - (1, 1, b, num_pos) -> (b, num_neg, b, num_pos)
        sim_matrix_diff = neg_pair_score[:, None, None] - pos_pair_score[None, None, :]
        dif = torch.cat([sim_matrix_diff.view(-1), torch.zeros(1, dtype=sim_matrix_diff.dtype, device=sim_matrix_diff.device)])
        loss = torch.logsumexp(dif, dim=0) / pos_pair_logits.size(0)
        return loss

    # def forward(self, pos_pair_logits: torch.Tensor, neg_pair_logits: torch.Tensor) -> torch.Tensor:
    #     pos_pair_score = torch.sigmoid(pos_pair_logits) / self.temperature  # (b, 1)
    #     neg_pair_score = torch.sigmoid(neg_pair_logits) / self.temperature  # (b, n_neg)

    #     sim_matrix_diff = neg_pair_score[None, :] - pos_pair_score[:, None]  # (b, b, n_neg)
    #     loss = torch.logsumexp(sim_matrix_diff, dim=-1).mean()
    #     # loss = torch.logsumexp(sim_matrix_diff)

    #     return loss


# Copy from https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py
class PairInBatchNegCoSentLoss(ContrastLoss):
    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor) -> torch.Tensor:
        """
        text_embddings: (batch_size, bedding_size)
        text_embeddings_pos: (batch_size, bedding_size)
        """
        # (batch_size, 1, 1, bedding_size) - (1, batch_size, num_pos, bedding_size) -> (batch_size, batch_size, num_pos)
        sim_matrix = torch.cosine_similarity(
            text_embeddings[:, None, None, :], text_embeddings_pos[None, :], dim=-1
        )  # (batch_size, batch_size, num_pos)
        # sim_matrix = sim_matrix / self.temperature
        sim_matrix_diag = torch.diagonal(sim_matrix).transpose(1, 0)
        sim_matrix_diff = sim_matrix - sim_matrix_diag[:, None, :] + self.margin
        sim_matrix_diff /= self.temperature
        dif = torch.cat(
            [
                sim_matrix_diff[~torch.eye(sim_matrix_diff.size(0)).bool()].view(-1),
                torch.zeros(1, dtype=sim_matrix_diff.dtype, device=sim_matrix_diff.device),
            ]
        )
        loss = torch.logsumexp(dif, dim=0) / text_embeddings.size(0)
        return loss

    # def forward(self, text_embeddings: torch.Tensor, text_pos_embeddings: torch.Tensor) -> torch.Tensor:
    #     """
    #     text_embddings: (batch_size, bedding_size)
    #     text_pos_embeddings: (batch_size, bedding_size)
    #     """
    #     sim_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_pos_embeddings.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
    #     sim_matrix = sim_matrix / self.temperature
    #     sim_matrix_diag = sim_matrix.diag()
    #     sim_matrix_diff = sim_matrix - sim_matrix_diag.unsqueeze(1)
    #     # sim_matrix_diag /= self.temperature
    #     loss = torch.logsumexp(sim_matrix_diff, dim=-1).mean()
    #     return loss


# Copy from https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py
class TripletInBatchNegCoSentLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, margin: float = 0, add_swap_loss: bool = False):
        super().__init__(temperature, margin)
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_softmax_loss = PairInBatchNegCoSentLoss(temperature)
        else:
            self._pair_contrast_softmax_loss = None

    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor, text_embeddings_neg: torch.Tensor) -> torch.Tensor:
        """
        text_embddings: (batch_size, bedding_size)
        text_embeddings_pos: (batch_size, bedding_size)
        text_embeddings_neg: (batch_size, bedding_size)
        """
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_embeddings_pos, dim=-1)
        sim_neg_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings_neg.unsqueeze(0), dim=-1)
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        sim_matrix_diff = sim_matrix - sim_matrix[:, 0].unsqueeze(1)
        loss = torch.logsumexp(sim_matrix_diff, dim=1).mean()
        if self.add_swap_loss:
            loss += self._pair_contrast_softmax_loss(text_embeddings_pos, text_embeddings)
        return loss


# Copy from https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py
class PairInBatchNegSigmoidContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, margin: float = 0):
        super().__init__(temperature, margin)
        self.temperature = temperature

    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor) -> torch.Tensor:
        """
        text_embddings: (batch_size, bedding_size)
        text_embeddings_pos: (batch_size, bedding_size)
        """
        batch_size = text_embeddings.size(0)
        # (batch_size, 1, embedding_size) @ (1, batch_size, embedding_size) -> (batch_size, batch_size)
        sim_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings_pos.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / self.temperature
        # (batch_size, batch_size) -> (batch_size,)
        sim_matrix_diag = sim_matrix.diag()

        # (batch_size, 1) - (batch_size, batch_size) -> (batch_size, batch_size)
        sim_diff_matrix = sim_matrix_diag.unsqueeze(1) - sim_matrix
        diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        # 对角线上的相似度分数差值是0，需要将其置为一个很大的数， 经过sigmoid后就趋近于1，损失要取log，进而相当于将这些位置的损失置0
        sim_diff_matrix = sim_diff_matrix.masked_fill(diag_mask, 1e9)

        # loss = -torch.log(torch.sigmoid(sim_diff_matrix)).sum() / (batch_size**2 - batch_size)
        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).sum() / (batch_size**2)
        return loss


# Copy from https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py
class TripletInBatchNegSigmoidContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, margin: float = 0, add_swap_loss: bool = False):
        super().__init__(temperature, margin)
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_sigmoid_loss = PairInBatchNegSigmoidContrastLoss(temperature)
        else:
            self._pair_contrast_sigmoid_loss = None

    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor, text_embeddings_neg: torch.Tensor) -> torch.Tensor:
        """
        text_embddings: (batch_size, bedding_size)
        text_embeddings_pos: (batch_size, bedding_size)
        text_embeddings_neg: (batch_size, bedding_size)
        """
        # (batch_size, embedding_size) @ (batch_size, embedding_size) -> (batch_size, )
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_embeddings_pos, dim=-1)
        sim_pos_vector = sim_pos_vector / self.temperature
        # (batch_size, 1, embedding_size) @ (1, batch_size, embedding_size) -> (batch_size, batch_size)
        sim_neg_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings_neg.unsqueeze(0), dim=-1)
        sim_neg_matrix = sim_neg_matrix / self.temperature
        # (batch_size, 1) - (batch_size, batch_size) -> (batch_size, batch_size)
        sim_diff_matrix = sim_pos_vector.unsqueeze(1) - sim_neg_matrix
        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).mean()
        if self.add_swap_loss:
            loss += self._pair_contrast_sigmoid_loss(text_embeddings_pos, text_embeddings)
        return loss


# Copy from https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py
# 该损失函数是论文https://aclanthology.org/2020.emnlp-main.550.pdf中损失函数的实现
class PairInBatchNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, margin: float = 0):
        super().__init__(temperature, margin)
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor) -> torch.Tensor:
        """
        text_embddings: (batch_size, bedding_size)
        text_embeddings_pos: (batch_size, bedding_size)
        """
        # (batch_size, 1, embedding_size) @ (1, batch_size, embedding_size) -> (batch_size, batch_size)
        sim_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings_pos.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long)
        loss = self.cross_entropy_loss(sim_matrix, labels)
        return loss


# Copy from https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py
class TripletInBatchNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, margin: float = 0, add_swap_loss: bool = False):
        super().__init__(temperature, margin)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_softmax_loss = PairInBatchNegSoftmaxContrastLoss(temperature)
        else:
            self._pair_contrast_softmax_loss = None

    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor, text_embeddings_neg: torch.Tensor) -> torch.Tensor:
        """
        text_embddings: (batch_size, embedding_size)
        text_embeddings_pos: (batch_size, embedding_size)
        text_embeddings_neg: (batch_size, embedding_size)
        text_embedding_pos是text_embeddings一一对应对应的pos,构成一个个正样本对.
        而text_embeddings_neg则是, 任何一个text_embeddings的与任何一个text_embeddings_neg都是负样本对. (inbatch就是体现在这里)
        """
        # (batch_size, embedding_size) @ (batch_size, embedding_size) -> (batch_size, )
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_embeddings_pos, dim=-1)
        # (batch_size, 1, embedding_size) @ (1, batch_size, embedding_size) -> (batch_size, batch_size)
        sim_neg_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings_neg.unsqueeze(0), dim=-1)
        # [(batch_size, 1) (batch_size, batch_size)] -> (batch_size, batch_size+1)
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        # -> (batch_size,)
        labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = self.cross_entropy_loss(sim_matrix, labels)
        if self._pair_contrast_softmax_loss:
            loss += self._pair_contrast_softmax_loss(text_embeddings_pos, text_embeddings)
        return loss


# 对TripletInBatchNegSoftmaxContrastLoss的改进
# 1. 可以接受一个正例和多个负例
# 2. inbatch改为用同batch内的其他样本正例作为当前样本的负例, 更合理
class TripletSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, margin: float = 0, add_inbatch_neg: bool = False):
        super().__init__(temperature, margin)
        self.add_inbatch_neg = add_inbatch_neg
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        if self.add_inbatch_neg:
            self.pair_inbatch_softmax_loss = PairInBatchNegSoftmaxContrastLoss(temperature)

    def forward(self, text_embeddings: torch.Tensor, text_embeddings_pos: torch.Tensor, text_embeddings_neg: torch.Tensor) -> torch.Tensor:
        """
        text_embddings: (batch_size, embedding_size)
        text_embeddings_pos: (batch_size, embedding_size)
        text_embeddings_neg: (batch_size, n, embedding_size)
        """
        # (batch_size, embedding_size) @ (batch_size, embedding_size) -> (batch_size, )
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_embeddings_pos, dim=-1)
        # (batch_size, 1, embedding_size) @ (batch_size, n, embedding_size) -> (batch_size, n)
        sim_neg_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings_neg, dim=-1)
        # [(batch_size, 1) (batch_size, n)] -> (batch_size, n+1)
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        # -> (batch_size,)
        labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = self.cross_entropy_loss(sim_matrix, labels)
        if self.add_inbatch_neg:
            loss += self.pair_inbatch_softmax_loss(text_embeddings, text_embeddings_pos)
        return loss


# # Copy from https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py
# class CoSentLoss(ContrastLoss):
#     bias: torch.Tensor

#     def __init__(self, temperature: float = 0.05) -> None:
#         super().__init__(temperature)
#         self.register_buffer("bias", torch.tensor([0.0]))

#     def forward(self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
#         predict_similarity = predict_similarity / self.temperature

#         cosine_similarity_diff = -(predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1))
#         smaller_mask = true_similarity.unsqueeze(0) <= true_similarity.unsqueeze(1)
#         cosine_similarity_diff = cosine_similarity_diff.masked_fill(smaller_mask, -1e12)

#         cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff.view(-1), self.bias))

#         loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0)
#         return loss
