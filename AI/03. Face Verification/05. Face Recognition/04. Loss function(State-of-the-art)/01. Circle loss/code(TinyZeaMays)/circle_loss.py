from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    """
        unsqueeze(idx)
            - 인덱스 위치에 1인 차원 추가
    """
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    """
        triu(diagonal=1)
            - 행렬의 대각요소 제어
            - diagonal=0 의 주 대각선 위의 모든 요소가 유지됨.
            - 음수 값은 대각선의 아래쪽, 양수는 위쪽으로 유지
    """
    """
        logical_not()
            - 주어진 입력 텐서의 요소별 논리 NOT 계산
            
            >>> torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
            tensor([ True, False, False])
    """
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    """
        view(-1)
            - 현재 차원의 실제 값 유추
    """
    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        
        """
            https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
            Softplus()
                - 1/β * log(1+exp(b*x))
                - params
                    beta – the β value for the Softplus formulation. Default: 1
                    threshold – values above this revert to a linear function. Default: 20
        """
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        """
            detach()
                - 기존 텐서에서 기울기 전파가 안되는 텐서 생성
                - storage를 공유하기에 생성한 텐서가 변경되면 원본 텐서도 변함.
            clamp()
                - 입력된 요소를 [min, max] 범위로 고정
        """
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        """
            logsumexp(x) = log ∑exp(x)
            https://pytorch.org/docs/stable/generated/torch.logsumexp.html
        """
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(256,))

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(inp_sp, inp_sn)

    print(circle_loss)