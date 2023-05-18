from re import template
import torch
import torch.nn as nn
from modules.bertembedding import BertEmbedding
from modules.tokenCL import TokenCL
from modules.rnn import LSTM
from modules.crf import CRF
from modules.attn import Attention
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from data.snips.generate_slu_emb import slot2desp
from utils.dataset import batch


class End2endSLUTagger(nn.Module):
    def __init__(self, vocabs, bert_embed_dim,
                 num_bound,
                 num_rnn_layer=1,
                 num_bert_layer=4,
                 dropout=0.0,
                 use_cl=False,
                 bert_model_path=None):
        super(End2endSLUTagger, self).__init__()
        self.vocabs = vocabs
        self.bert_embed_dim = bert_embed_dim
        self.num_bound = num_bound
        self.dropout = dropout

        self.bert = BertEmbedding(bert_model_path,
                                #   num_bert_layer,
                                  merge='none',
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=False)

        hidden_size = self.bert_embed_dim // 2
        self.seq_encoder = LSTM(input_size=self.bert_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer)

        self.bound_embedding = nn.Embedding(num_embeddings=num_bound,
                                            embedding_dim=10,
                                            padding_idx=0)
        nn.init.xavier_uniform_(self.bound_embedding.weight)

        self.hidden2boundary = nn.Linear(2*hidden_size, num_bound)

        self.tag_crf = CRF(num_tags=num_bound, batch_first=True)

        self.temperature = 1.0
        self.cl_temperature = 1.0
        self.hidden_proj = nn.Linear(2 * hidden_size + 10, 768)

        # self.label_proj = nn.Sequential(
        #     nn.Linear(2*hidden_size, hidden_size//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size//2, 2*hidden_size)
        # )

        self.use_cl = use_cl
        if use_cl:
            self.tokencl = TokenCL(self.bert_embed_dim)
        # self.slu_encoder = LSTM(input_size=2 * hidden_size + 10,
        #                         hidden_size=hidden_size,
        #                         num_layers=num_rnn_layer,
        #                         dropout=dropout)
        
        # TODO: self.vocabs['type']
        # self.label_proj = nn.Linear(768, 768, bias=False)
        
        
        # self.proj_slot_emb1 = nn.Linear(768, 768//2, bias=False)
        # self.proj_slot_emb2 = nn.Linear(768//2, 768, bias=False)
        # self.proj_slot_emb = nn.Sequential(
        #     self.proj_slot_emb1,
        #     nn.Tanh(),
        #     self.proj_slot_emb2
        # )
        
    def bert_params(self):
        return self.bert.bert.parameters()
        # return self.bert.parameters()

    def bert_named_params(self):
        return self.bert.bert.named_parameters()
        # return self.bert.named_parameters()

    def base_params(self):
        bert_param_names = list(map(id, self.bert.bert.parameters()))
        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append(param)
        return other_params

    def base_named_params(self):
        bert_param_names = list(map(id, self.bert.bert.parameters()))
        # cl_param_names = list(map(id, self.tokencl.parameters()))
        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append((name, param))
        return other_params
    
    def cl_named_params(self):
        return self.tokencl.named_parameters()

    # TODO: KL loss
    def get_smooth_label(self, type_labels):
        onehot_labels = F.one_hot(type_labels, num_classes=len(self.vocabs["slot"]))
        bsz, seq_len, num_labels = onehot_labels.size()

        # proj_label_emb = self.proj_slot_emb(self.label_embedding)
        slots_similarity = (self.label_embedding @ self.label_embedding.T) # / (torch.linalg.norm(self.label_embedding, dim=-1)**2)
        slot_labels = onehot_labels + slots_similarity.index_select(dim=0, index=type_labels.view(-1)).view(bsz, seq_len, -1)
        return F.softmax(onehot_labels + slot_labels, dim=-1)


    def evaluate(self, bert_inp, num_type, mask=None):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        '''
        bert_repr = self.bert(*bert_inp)

        label_embedding = bert_repr[:, 1: num_type+1]
        token_repr = bert_repr[:, num_type+1:]

        enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        boundary_score = self.hidden2boundary(enc_out)

        pred_bound = self.tag_decode(boundary_score, mask)

        bound_embed = self.bound_embedding(pred_bound)

        type_logits = self.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
        type_logits = F.normalize(type_logits, p=2, dim=-1)
        label_embedding = F.normalize(label_embedding, p=2, dim=-1)
        type_score = torch.matmul(type_logits, label_embedding.transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
        
        pred_type = type_score.data.argmax(dim=-1)
        return pred_bound, pred_type

    def nt_xent(self, loss, num, denom, temperature=1):
        "cross-entropy loss for CL"
        loss = torch.exp(loss / temperature)
        cnts = torch.sum(num, dim = 1)
        loss_num = torch.sum(loss * num, dim = 1)
        loss_denom = torch.sum(loss * denom, dim = 1)
        # sanity check
        nonzero_indexes = torch.where(cnts > 0)
        loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]

        loss_final = -torch.log(loss_num) + torch.log(loss_denom)
        return loss_final

    def forward(self, bert_inp, num_type, boundary_labels, type_labels, O_tag_idx, mask=None):
        ''' boundary tags -> type tags: B I E S  -> LOC PER ...
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        bert_repr = self.bert(*bert_inp)

        label_embedding = bert_repr[:, 1: num_type+1]
        token_repr = bert_repr[:, num_type+1:]

        token_repr = F.dropout(token_repr, p=self.dropout, training=self.training)

        enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)

        boundary_score = self.hidden2boundary(enc_out)
        boundary_loss = self.tag_loss(boundary_score, boundary_labels, mask, reduction='none')
         
        # differentiable
        bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), self.bound_embedding.weight)

        type_logits = self.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())

        # # regular cross entropy loss
        # type_logits = F.normalize(type_logits, p=2, dim=-1)
        # label_embedding = F.normalize(label_embedding, p=2, dim=-1)
        # type_score = torch.matmul(type_logits, label_embedding.transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)

        # type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        # type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        # type_loss = torch.sum(type_loss * type_mask, dim=1)
        # loss = torch.mean(boundary_loss + type_loss)

        # info-nce loss
        bsz, seq_len, _ = type_logits.size()
        type_logits = F.normalize(type_logits, p=2, dim=-1)
        label_embedding = F.normalize(label_embedding, p=2, dim=-1)
        # (B, N, D) * (B, D, K) -> (B * N, K)
        loss_cos = torch.matmul(type_logits, label_embedding.transpose(1, 2).contiguous()).view(bsz * seq_len, num_type)

        denominator = torch.ones((bsz * seq_len, num_type), device=type_labels.device).float()
        numerator = F.one_hot(type_labels).view(bsz * seq_len, num_type).float()
        type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        type_loss = torch.sum(self.nt_xent(loss_cos, numerator, denominator, temperature=0.1).view(bsz, seq_len) * type_mask, dim=1)
        loss = torch.mean(boundary_loss + type_loss)

        # add orthogonal regression
        bsz, _, _ = label_embedding.size()
        loss += F.mse_loss(label_embedding @ label_embedding.transpose(-1, -2), 
                            torch.eye(label_embedding.size(1), device=label_embedding.device).unsqueeze(0).expand(bsz, num_type, num_type))
        if self.use_cl:
            cl_loss = self.tokencl(token_repr, type_labels, mask)
            loss += cl_loss
        else:
            cl_loss = torch.tensor([0.])
            
        return loss, {"bio_loss": boundary_loss.mean().item(), "slu_loss": type_loss.mean().item(), "cl_loss": cl_loss.item()}

    
    def tag_loss(self, tag_score, gold_tags, mask=None, reduction='mean', alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            lld = self.tag_crf(tag_score, tags=gold_tags, mask=mask, reduction=reduction)
            return lld.neg()
        else:
            sum_loss = F.cross_entropy(tag_score.transpose(1, 2), gold_tags, ignore_index=0, reduction='sum')
            return sum_loss / mask.sum()

    def tag_decode(self, tag_score, mask=None, alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)  emission probs
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg:
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            best_tag_seq = self.tag_crf.decode(tag_score, mask=mask)
            # return best segment tags
            return pad_sequence(best_tag_seq, batch_first=True, padding_value=0)
        else:
            return tag_score.data.argmax(dim=-1) * mask.long()
