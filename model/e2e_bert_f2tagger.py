from audioop import bias
import torch
import torch.nn as nn
from modules.bertembedding import BertEmbedding
from modules.tokenCL import TokenCL
from modules.rnn import LSTM
from modules.crf import CRF
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math


def js_div(p_output, q_output, p_mask, q_mask):
    """
    Function that measures JS divergence between target and output logits:
    """
    p_output = F.softmax(p_output, dim=-1)
    q_output = F.softmax(q_output, dim=-1)
    log_mean_output = torch.log((p_output + q_output) / 2)
    # return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
    return F.kl_div(log_mean_output, p_output, reduction='none') * p_mask.unsqueeze(-1) + \
           F.kl_div(log_mean_output, q_output, reduction='none') * q_mask.unsqueeze(-1)
           

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index, reduction, label_smoothing):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.lb_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, seen_logits, unseen_logits, label, similarity):
        """
        Args:
            logits (_type_): size(B, N, num_type)
            label (_type_): size(B, N)
            mask (_type_): size(B, N)
        Returns:
            _type_: smoothing loss
        """
        bsz, seq_len = label.size()
        num_seen_classes = seen_logits.size(-1)
        num_unseen_classes = unseen_logits.size(-1)
        if self.ignore_index is not None:
            ignore = label.eq(self.ignore_index)
        lb_one_hot = F.one_hot(label, num_classes=num_seen_classes) * (1. - self.lb_smoothing)

        unseen_lb = torch.gather(similarity, 1, label.unsqueeze(-1).expand(bsz, seq_len, num_unseen_classes))
        logits = torch.cat([seen_logits, unseen_logits], dim=-1)
        sm_label = torch.cat([lb_one_hot, unseen_lb * self.lb_smoothing], dim=-1)
        
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * sm_label, dim=-1)
        loss[ignore] = 0
        if self.reduction == 'batchmean':
            loss = loss.sum() / logits.size(0)
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


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
        self.hidden_proj = nn.Linear(2 * hidden_size + 10, self.bert_embed_dim)

        self.use_cl = use_cl
        if use_cl:
            self.tokencl = TokenCL(self.bert_embed_dim, cl_type, cl_temperature)
        
        self.ce_loss = LabelSmoothingCrossEntropyLoss(ignore_index=0, reduction='none', label_smoothing=0.2)
        # TODO: ReLU, GELU
        self.label_adapter = nn.Sequential(nn.Linear(self.bert_embed_dim, self.bert_embed_dim//4, bias=False),
                                            getattr(nn, 'GELU')(),
                                            # nn.Dropout(p=0.3),
                                            nn.Linear(self.bert_embed_dim//4, self.bert_embed_dim, bias=False))
        # std initialization for adapter
        # for m in self.label_adapter:
        #     if type(m) == nn.Linear:
        #         nn.init.normal_(m.weight, std=0.02)
        # TODO: LoRA
        # self.lora_A = nn.Parameter(self.hidden_proj.weight.new_zeros((self.bert_embed_dim//4, self.bert_embed_dim)))
        # self.lora_B = nn.Parameter(self.hidden_proj.weight.new_zeros((self.bert_embed_dim, self.bert_embed_dim//4)))
        # # self.scaling = nn.Parameter(torch.ones(1))
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B)
        # # self.lnorm = nn.LayerNorm(self.bert_embed_dim, eps=1e-12)
        # TODO: VAE
        # self.latent_size = self.bert_embed_dim // 6
        # self.hidden2mean = nn.Linear(self.bert_embed_dim, self.latent_size)
        # self.hidden2logv = nn.Linear(self.bert_embed_dim, self.latent_size)
        # self.latent2hidden = nn.Linear(self.latent_size, self.bert_embed_dim)

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

    def evaluate(self, bert_inp, num_type, mask=None, tsne=False):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        '''
        bert_repr = self.bert(*bert_inp)
        # label_embedding = bert_repr[:, 1: num_type+1]
        # TODO: adapter
        label_embedding = self.label_adapter(bert_repr[:, 1: num_type+1]) + bert_repr[:, 1: num_type+1]
        # TODO: LoRA
        # label_embedding = (bert_repr[:, 1: num_type+1].detach() @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) + bert_repr[:, 1: num_type+1].detach()
        # TODO: VAE
        # (bsz, seq_len, latent_size)
        # mean = self.hidden2mean(bert_repr[:, 1: num_type+1])
        # logv = self.hidden2logv(bert_repr[:, 1: num_type+1])
        # std = torch.exp(0.5 * logv)
        # z = torch.randn([bert_repr.size(0), num_type, self.latent_size], device=bert_repr.device)
        # z = z * std + mean
        # label_embedding = self.latent2hidden(z)
        
        token_repr = bert_repr[:, num_type+1:]
        
        # bert_repr = self.bert(*bert_inp)
        # label_embedding = bert_repr[:, 4: num_type+4]
        # token_repr = bert_repr[:, num_type+4:]

        enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        boundary_score = self.hidden2boundary(enc_out)

        pred_bound = self.tag_decode(boundary_score, mask)
        bound_embed = self.bound_embedding(pred_bound)

        type_logits = self.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
        type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
        
        pred_type = type_score.data.argmax(dim=-1)
        if not tsne:
            return pred_bound, pred_type
        else:
            return F.normalize(type_logits, p=2, dim=-1).detach(), F.normalize(label_embedding, p=2, dim=-1).detach(), pred_type

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

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)
    
    def forward(self, bert_inp, num_type, boundary_labels, 
                type_labels, O_tag_idx, mask=None, train='utter', step=0, x0=0):
        ''' boundary tags -> type tags: B I E S  -> LOC PER ...
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        
        bert_repr = self.bert(*bert_inp)
        # label_embedding = bert_repr[:, 1: num_type+1]
        # TODO: Adapter
        label_embedding = self.label_adapter(bert_repr[:, 1: num_type+1]) + bert_repr[:, 1: num_type+1]
        # TODO: LoRA
        # label_embedding = (bert_repr[:, 1: num_type+1].detach() @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) + bert_repr[:, 1: num_type+1].detach()
        # TODO: VAE
        # (bsz, num_type, latent_size)
        # mean = self.hidden2mean(bert_repr[:, 1: num_type+1])
        # logv = self.hidden2logv(bert_repr[:, 1: num_type+1])
        # std = torch.exp(0.5 * logv)
        # z = torch.randn([bert_repr.size(0), num_type, self.latent_size], device=bert_repr.device)
        # z = z * std + mean
        # label_embedding = self.latent2hidden(z)
        # # TODO: KL_weight * KL_loss
        # KL_weight = self.kl_anneal_function(anneal_function='linear', step=step, k=0.0025, x0=x0)
        # KL_loss = -0.5 * KL_weight * torch.sum(1 + logv - mean.pow(2) - logv.exp(), dim=-1).sum() / bert_repr.size(0)
        
        token_repr = bert_repr[:, num_type+1:]
        
        # bert_repr = self.bert(*bert_inp)
        # label_embedding = bert_repr[:, 4: num_type+4]
        # token_repr = bert_repr[:, num_type+4:]
        # TODO: uttr+label cl
        # cl_repr = bert_repr[:, 1:]
        
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
        
        # type_score_utter = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2).detach()) / self.temperature  # (B, N, D) x (B, D, K)
        # type_loss_utter = F.cross_entropy(type_score_utter.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        # type_loss = 0.1 * type_loss_label + type_loss_utter
        
        type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        type_loss = torch.sum(type_loss * type_mask, dim=1)
        
        # TODO: ablation study
        # type_loss = torch.zeros_like(boundary_loss, device=token_repr.device)
        # TODO: VAE KL_loss
        loss = torch.mean(boundary_loss + type_loss)
        # elif train == 'label':
        #     loss = torch.mean(boundary_loss + type_loss) + KL_loss

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
        # if train == 'label':
        #     normalized_label_embedding = F.normalize(label_embedding-label_embedding.mean(-1).unsqueeze(-1), p=2, dim=-1)
        #     bsz = normalized_label_embedding.size(0)
        #     orth_loss = F.mse_loss(normalized_label_embedding @ normalized_label_embedding.transpose(-1, -2).contiguous(),
        #                         torch.eye(label_embedding.size(1), device=normalized_label_embedding.device).unsqueeze(0).expand(bsz, num_type, num_type))
        #     diag_mask = 1 - torch.eye(label_embedding.size(1), device=label_embedding.device).unsqueeze(0).expand(label_embedding.size(0), num_type, num_type)
        #     masked_orth_loss = (orth_loss * diag_mask).sum() / diag_mask.sum()
        #     loss += 0.8 * masked_orth_loss
        # else:
        masked_orth_loss = torch.tensor([0.], device=token_repr.device)
        
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
        
        # # version.4 feature space orthogonal
        # _embedding = label_embedding.mean(0).mean(0)
        # dot_prod = F.normalize(_embedding.unsqueeze(0), p=2, dim=-1).transpose(-1, -2) @ F.normalize(_embedding.unsqueeze(0), p=2, dim=-1).contiguous()
        # diag = torch.eye(dot_prod.size(1), device=label_embedding.device)
        # orth_loss = F.l1_loss(dot_prod, diag, reduction='none').mean()
        # loss += orth_loss
        # orth_loss = torch.tensor([0.], device=token_repr.device)
        
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
        if self.use_cl and train == 'utter':
            cl_loss = self.tokencl(token_repr, type_labels, mask)
            # TODO: uttr+label CL.
            # cl_loss = self.tokencl(cl_repr, cl_labels,
            #                        torch.cat([torch.ones((cl_repr.size(0), num_type), device=cl_repr.device), 
            #                                   mask], dim=1))
            loss += cl_loss
        else:
            cl_loss = torch.tensor([0.])

        # return loss, {"bio_loss": boundary_loss.mean().item(), "slu_loss": type_loss.mean().item(), "cl_loss": cl_loss.item(), "orth_loss": orth_loss, "utter": torch.sum(type_loss_utter * type_mask, dim=1).mean().item(), "label": torch.sum(type_loss_label * type_mask, dim=1).mean().item()}
        return loss, {"bio_loss": boundary_loss.mean().item(), "slu_loss": type_loss.mean().item(), "cl_loss": cl_loss.item(), "orth_loss": masked_orth_loss}
    
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
