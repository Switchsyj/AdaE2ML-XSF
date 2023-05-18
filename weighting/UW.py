import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weighting.abstract_weighting import AbsWeighting

class UW(AbsWeighting):
    r"""Uncertainty Weights (UW).
    
    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_ \
    and implemented by us. 
    """
    def __init__(self,
                 task_name,
                 device):
        super(UW, self).__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.init_param()
        
    def init_param(self):
        # the first loss is main loss.
        self.loss_scale = nn.Parameter(torch.tensor([0.5], device=self.device), requires_grad=True)
        
    def backward(self, losses):
        # loss = losses[0] + (losses[1:] / (2*self.loss_scale.exp()) + self.loss_scale / 2).sum()
        loss = losses[0] + losses[1] * self.loss_scale + losses[2] * (1 - self.loss_scale)
        loss.backward()
        return (1 / (2*torch.exp(self.loss_scale))).detach().cpu().numpy()