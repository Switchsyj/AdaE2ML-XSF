import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceCL(nn.Module):
    def __init__(self, temperature=0.01):
        super(InstanceCL, self).__init__()
        self.temperature = temperature
    
    def forward(self, sent_reprs, mask=None):
        bsz, seq_len, _ = sent_reprs.size()
        sent_reprs_b = F.normalize(F.dropout(sent_reprs, p=0.1) + 1e-12, dim=-1)
        sent_reprs = F.normalize(sent_reprs + 1e-12, dim=-1)
        # (bsz, len, len)
        cos_sim = (torch.bmm(sent_reprs, sent_reprs_b.transpose(1, 2))) / \
                    (torch.matmul(torch.linalg.norm(sent_reprs, dim=-1).unsqueeze(2), torch.linalg.norm(sent_reprs_b, dim=-1).unsqueeze(1)) + 1e-12)
        
        # (bsz, len)
        numerator = torch.exp(torch.sum(torch.matmul(cos_sim, torch.eye(seq_len, seq_len).unsqueeze(0).to(sent_reprs.device)), dim=-1) / self.temperature)
        # (bsz, len)
        denominator = torch.exp(torch.sum(cos_sim, dim=-1) / self.temperature)
        return torch.sum(-torch.log(numerator / (denominator + 1e-12)), dim=1).mean()
        


