import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weighting.abstract_weighting import AbsWeighting

class DWA(AbsWeighting):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 
    Args:
        T (float, default=2.0): The softmax temperature.
    """
    def __init__(self,
                task_name,
                device
                ):
        super(DWA, self).__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.train_loss_buffer = []
        self.epoch = 0
        
    def backward(self, losses, T=1.0):
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[self.epoch-1] / self.train_loss_buffer[self.epoch-2]).to(self.device)
            batch_weight = self.task_num * F.softmax(w_i / T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight.unsqueeze(1)).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()