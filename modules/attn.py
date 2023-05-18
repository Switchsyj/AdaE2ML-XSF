import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.input_proj = nn.Linear(input_size, 1, bias=False)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.uniform_(self.input_proj.weight.data, -0.01, 0.01)
        
    def forward(self, input, mask):
        logits = self.input_proj(input)
        logits = torch.exp(logits - logits.max()) * mask.unsqueeze(-1)
        denominator = logits.sum(dim=1, keepdim=True)
        attn_score = (logits / (denominator)).expand_as(input)
        # (bsz, hidden_size)
        return torch.mul(input, attn_score).sum(dim=1)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        bsz = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(bsz, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        attn_score = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(attn_score)
            attn_score = attn_score.masked_fill(mask==0, -1e9)
        attn_score = F.softmax(attn_score, dim=-1)
        if self.dropout > 0.:
            attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        output = torch.matmul(attn_score, value)

        # 3) "Concat" using a view and apply a final linear.
        output = output.transpose(1, 2).contiguous().view(bsz, -1, self.h * self.d_k)

        del query
        del key
        del value
        return self.linears[-1](output)