from audioop import bias
import torch
import torch.nn as nn
from modules.bertembedding import BertEmbedding
from modules.embedding import PretrainedEmbedding
from modules.tokenCL import TokenCL
from modules.rnn import LSTM
from modules.crf import CRF
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np       


class End2endSLUTagger(nn.Module):
    def __init__(self, vocabs, bert_embed_dim,
                 num_bound,
                 num_rnn_layer=1,
                 num_bert_layer=4,
                 dropout=0.0,
                 use_cl=False,
                 cl_type='cosine',
                 bert_model_path=None,
                 cl_temperature=1.0
                 ):
        super(End2endSLUTagger, self).__init__()
        self.vocabs = vocabs
        self.bert_embed_dim = bert_embed_dim
        self.num_bound = num_bound
        self.dropout = dropout

        self.bert = BertEmbedding(model_path=bert_model_path,
                                  merge='none',
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=False)
        self.pretrained_embedding = PretrainedEmbedding(
            self.vocabs['token'], 
            emb_file='/cognitive_comp/shiyuanjun/zero-shot-slu/data/snips/cache/glove.840B.300d.txt', 
            num_words=len(self.vocabs['token']), 
            emb_dim=300
        )

        hidden_size = self.bert_embed_dim // 2
        self.seq_encoder = LSTM(input_size=self.bert_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer)

        self.bound_embedding = nn.Embedding(num_embeddings=num_bound,
                                            embedding_dim=10,
                                            padding_idx=0)
        nn.init.xavier_uniform_(self.bound_embedding.weight)

        self.hidden2boundary = nn.Linear(2*hidden_size, num_bound)
        self.down_proj = nn.Linear(1068, 768)
        # self.up_proj = nn.Linear(300, 768)

        self.tag_crf = CRF(num_tags=num_bound, batch_first=True)
        self.temperature = 1.0
        self.hidden_proj = nn.Linear(2 * hidden_size + 10, self.bert_embed_dim)

        self.use_cl = use_cl
        if use_cl:
            self.tokencl = TokenCL(self.bert_embed_dim, cl_type, cl_temperature)
        
        self.label_adapter = nn.Sequential(nn.Linear(self.bert_embed_dim, self.bert_embed_dim//4, bias=False),
                                    getattr(nn, 'GELU')(),
                                    # nn.Dropout(p=0.3),
                                    nn.Linear(self.bert_embed_dim//4, self.bert_embed_dim, bias=False))

    def bert_params(self):
        return self.bert.bert.parameters()

    def bert_named_params(self):
        return self.bert.bert.named_parameters()
    
    def base_params(self):
        bert_param_names = list(map(id, self.bert_params()))
        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append(param)
        return other_params

    def base_named_params(self):
        bert_param_names = list(map(id, self.bert_params()))
        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append((name, param))
        return other_params

    def evaluate(self, bert_token_inps, bert_label_inps, glove_label_inps, num_type, mask=None):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        ''' 
        # bert embedding
        bert_embedding = self.bert(*bert_token_inps)
        bert_label_repr = bert_embedding[:, 1: num_type+1]
        bert_token_repr = bert_embedding[:, num_type+1:]
        # bert_label_repr = self.bert(*bert_label_inps)[:, 1:].squeeze(1)
        # # bert_label_repr = bert_label_repr.expand(bert_token_repr.size(0), bert_label_repr.size(1), bert_label_repr.size(2)).contiguous()
        # bert_label_repr = bert_label_repr.unsqueeze(0).expand(bert_token_repr.size(0), bert_label_repr.size(0), bert_label_repr.size(1)).contiguous()
        
        # backbone
        # bert_repr = self.bert(*bert_token_inps)
        # token_repr = bert_repr[:, num_type+1:]
        # bert_label_repr = bert_repr[:, 1: num_type+1]
        
        # pretrained embedding
        glove_label_repr = self.pretrained_embedding(glove_label_inps[0])
        label_chunks = glove_label_repr.view(-1, glove_label_repr.size(-1)).split(glove_label_inps[1].view(-1).tolist())
        glove_label_embedding = torch.stack(tuple([bc.mean(0) for bc in label_chunks])).view(glove_label_inps[1].size(0), glove_label_inps[1].size(1), -1).contiguous()
        
        # label_embedding = self.up_proj(glove_label_embedding)
        # label_embedding = label_embedding + self.label_adapter(label_embedding)
        
        # TODO: discrete embedding.
        # bert_label_repr = []
        # for _, inps in enumerate(bert_label_inps):
        #     inps = [x.to(bert_token_repr.device) for x in inps]
        #     bert_label_repr.append(self.bert(*inps)[:, 1:])
        # bert_label_repr = torch.cat(bert_label_repr, dim=1).expand(bert_token_repr.size(0), len(bert_label_repr), bert_label_repr[0].size(-1)).contiguous()
        # label_embedding = label_embedding + self.label_adapter(label_embedding)
        label_embedding = torch.cat([bert_label_repr, glove_label_embedding], dim=-1)
        label_embedding = self.down_proj(label_embedding)
        label_embedding = self.label_adapter(label_embedding) + label_embedding
        
        token_repr = bert_token_repr
        # label_embedding = bert_label_repr

        enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        boundary_score = self.hidden2boundary(enc_out)

        pred_bound = self.tag_decode(boundary_score, mask)
        bound_embed = self.bound_embedding(pred_bound)

        type_logits = self.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
        type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
        
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

    def forward(self, bert_token_inps, bert_label_inps, glove_label_inps, 
                num_type, boundary_labels, 
                type_labels, O_tag_idx, mask=None, train='utter'):
        ''' boundary tags -> type tags: B I E S  -> LOC PER ...
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        # bert embedding
        bert_embedding = self.bert(*bert_token_inps)
        bert_label_repr = bert_embedding[:, 1: num_type+1]
        bert_token_repr = bert_embedding[:, num_type+1:]
        # bert_label_repr = self.bert(*bert_label_inps)[:, 1:].squeeze(1)
        # bert_label_repr = bert_label_repr.expand(bert_token_repr.size(0), bert_label_repr.size(1), bert_label_repr.size(2)).contiguous()
        # bert_label_repr = bert_label_repr.unsqueeze(0).expand(bert_token_repr.size(0), bert_label_repr.size(0), bert_label_repr.size(1)).contiguous()
        
        # backbone
        # bert_repr = self.bert(*bert_token_inps)
        # token_repr = bert_repr[:, num_type+1:]
        # bert_label_repr = bert_repr[:, 1: num_type+1]
        
        # pretrained embedding
        glove_label_repr = self.pretrained_embedding(glove_label_inps[0])
        label_chunks = glove_label_repr.view(-1, glove_label_repr.size(-1)).split(glove_label_inps[1].view(-1).tolist())
        glove_label_embedding = torch.stack(tuple([bc.mean(0) for bc in label_chunks])).view(glove_label_inps[1].size(0), glove_label_inps[1].size(1), -1).contiguous()
        
        # TODO: discrete embedding.
        # bert_label_repr = []
        # for _, inps in enumerate(bert_label_inps):
        #     inps = [x.to(bert_token_repr.device) for x in inps]
        #     bert_label_repr.append(self.bert(*inps)[:, 1:])
        # bert_label_repr = torch.cat(bert_label_repr, dim=1).expand(bert_token_repr.size(0), len(bert_label_repr), bert_label_repr[0].size(-1)).contiguous()
        # label_embedding = (label_embedding + self.label_adapter(label_embedding)).detach()
        
        label_embedding = torch.cat([bert_label_repr, glove_label_embedding], dim=-1)
        label_embedding = self.down_proj(label_embedding)
        label_embedding = self.label_adapter(label_embedding) + label_embedding
        # label_embedding = self.up_proj(glove_label_embedding)
        
        token_repr = bert_token_repr
        # label_embedding = bert_label_repr
        
        # if num_unseen_type > 0:
        #     unseen_label_embedding = self.bert(*unseen_inp)[:, 1:]
        # else:
        #     unseen_label_embedding = None
        
        token_repr = F.dropout(token_repr, p=self.dropout, training=self.training)
        enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)

        boundary_score = self.hidden2boundary(enc_out)
        boundary_loss = self.tag_loss(boundary_score, boundary_labels, mask, reduction='none')
         
        # differentiable
        bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), self.bound_embedding.weight)
        type_logits = self.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
        
        if train == 'utter':
            type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2).detach()) / self.temperature  # (B, N, D) x (B, D, K)
            type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        elif train == 'label':
            type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1).detach(), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
            type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        # type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
        # type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        type_loss = torch.sum(type_loss * type_mask, dim=1)
        loss = torch.mean(boundary_loss + type_loss)

        # # info-nce loss
        # bsz, seq_len, _ = type_logits.size()
        # # (B, N, D) * (B, D, K) -> (B * N, K)
        # loss_cos = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(1, 2).contiguous()).view(bsz * seq_len, num_type)
        # # (B, N, K) - margin
        # # loss_cos = (loss_cos - F.one_hot(type_labels, num_classes=num_type) * 0.05).view(bsz * seq_len, num_type)

        # denominator = torch.ones((bsz * seq_len, num_type), device=type_labels.device).float()
        # numerator = F.one_hot(type_labels, num_classes=num_type).view(bsz * seq_len, num_type).float()
        # type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        # type_loss = torch.sum(self.nt_xent(loss_cos, numerator, denominator, temperature=0.5).view(bsz, seq_len) * type_mask, dim=1)
        # loss = torch.mean(boundary_loss + type_loss)
        
        # # add normalized orthogonal regression
        # normalized_label_embedding = F.normalize(label_embedding-label_embedding.mean(-1).unsqueeze(-1), p=2, dim=-1)
        # bsz = normalized_label_embedding.size(0)
        # orth_loss = F.mse_loss(normalized_label_embedding @ normalized_label_embedding.transpose(-1, -2).contiguous(),
        #                        torch.eye(label_embedding.size(1), device=normalized_label_embedding.device).unsqueeze(0).expand(bsz, num_type, num_type))
        # diag_mask = 1 - torch.eye(label_embedding.size(1), device=label_embedding.device).unsqueeze(0).expand(label_embedding.size(0), num_type, num_type)
        # loss += (orth_loss * diag_mask).sum() / diag_mask.sum()
        
        # (type, 1, 768) - (1, type, 768) -> (type, tpye, 768)
        # euc_loss = (F.normalize(label_embedding.mean(0), p=2, dim=-1).unsqueeze(1) - F.normalize(label_embedding.mean(0), p=2, dim=-1).unsqueeze(0)) ** 2
        # diag_mask = (1 - torch.eye(label_embedding.size(1), device=label_embedding.device)).unsqueeze(-1)
        # euc_loss = torch.mean(euc_loss * diag_mask, dim=-1).sum() / diag_mask.sum()

        # cos_loss = 1 - F.normalize(label_embedding, p=2, dim=-1) @ F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)
        # diag_mask = 1 - torch.eye(label_embedding.size(1), device=label_embedding.device).unsqueeze(0).expand(label_embedding.size(0), num_type, num_type)
        # cos_loss = (cos_loss * diag_mask).sum() / diag_mask.sum()
        
        # TODO: range from (0-1)
        # margin_loss = torch.log(torch.abs(euc_loss - 0.7) + 1.0)
        # loss += euc_loss
        
        # bsz, _, _ = label_embedding.size()
        # orth_loss = F.mse_loss(F.normalize(label_embedding, p=2, dim=-1) @ F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2),
        #                        torch.eye(label_embedding.size(1), device=label_embedding.device).unsqueeze(0).expand(bsz, num_type, num_type))
        # orth_mask = 1 - torch.eye(label_embedding.size(1), device=label_embedding.device).unsqueeze(0).expand(bsz, num_type, num_type)
        # orth_loss = F.l1_loss(F.normalize(label_embedding, p=2, dim=-1) @ F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2),
        #                       torch.eye(label_embedding.size(1), device=label_embedding.device).unsqueeze(0).expand(bsz, num_type, num_type), 
        #                       reduction='none') * orth_mask
        # orth_loss = torch.mean(orth_loss) * 0.5
        # loss += orth_loss

        # # version.2 orthogonal regression
        # dot_prod = F.normalize(label_embedding, p=2, dim=-1) @ F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2).contiguous()
        # ones = torch.ones(dot_prod.shape, device=label_embedding.device) 
        # diag = torch.eye(dot_prod.size(1), device=label_embedding.device).unsqueeze(0).expand_as(ones)
        # loss += ((dot_prod * (ones - diag))**2).sum() / label_embedding.size(0)

        # # version.3 orthogonal div
        # dot_prod = F.normalize(label_embedding, p=2, dim=-1) @ F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2).contiguous()
        # dot_prod = F.softmax(dot_prod, dim=-1)
        # # KL(P||Q)=Q(logQ-logP), P is pred, Q is gold
        # loss += F.kl_div(dot_prod.log(), F.softmax(torch.eye(label_embedding.size(1), device=label_embedding.device), dim=-1).unsqueeze(0).expand_as(dot_prod), reduction='batchmean')
        
        # # use unseen label entropy regularization
        # if num_unseen_type > 0:
        #     # regular p log p
        #     # (batch, seq_len, unseen)
        #     unseen_label_embedding = self.mh_attn(unseen_label_embedding, label_embedding, label_embedding)
        #     unseen_type_score = F.softmax(torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(unseen_label_embedding, p=2, dim=-1).transpose(-1, -2)), dim=-1)
        #     # qc(x_n) * log qc(x_n)
        #     unseen_loss = (-unseen_type_score * torch.log(unseen_type_score)).sum(2)
        #     unseen_loss = 0.1 * torch.mean((unseen_loss * type_mask).sum(1))
        #     loss += unseen_loss

        #     # # supervised unseen target
        #     # # (batch, seq_len, unseen)
        #     # unseen_label_embedding = self.mh_attn(label_embedding, unseen_label_embedding, unseen_label_embedding)
        #     # unseen_type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(unseen_label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature
        #     # # qc(x_n) * log qc(x_n)
        #     # unseen_loss = F.cross_entropy(unseen_type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        #     # unseen_loss = 0.1 * torch.sum(unseen_loss * type_mask, dim=1)
        #     # loss += torch.mean(unseen_loss)

        #     # # pseudo label
        #     # # # (batch, seq_len, unseen)
        #     # unseen_type_score = F.softmax(torch.matmul(type_logits, unseen_label_embedding.detach().transpose(-1, -2)) / self.temperature, dim=-1)
        #     # # (batch, seen, unseen)
        #     # label_sim = torch.bmm(F.normalize(label_embedding, p=2, dim=-1), F.normalize(unseen_label_embedding, p=2, dim=-1).transpose(-1, -2))
        #     # # (batch, seen)
        #     # label_sim_idx = torch.argmax(label_sim, dim=-1)
        #     # unseen_label = label_sim_idx.gather(dim=1, index=type_labels)
        #     # # (batch, seq_len, 1)
        #     # unseen_loss = F.cross_entropy(unseen_type_score.transpose(1, 2).contiguous(), unseen_label, reduction='none')
        #     # unseen_loss = torch.sum(unseen_loss * type_mask, dim=1)
        #     # loss += 0.1 * torch.mean(unseen_loss)

        # # else:
        # unseen_loss = torch.tensor([0.], device=type_labels.device)

        # use tokenCL
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
