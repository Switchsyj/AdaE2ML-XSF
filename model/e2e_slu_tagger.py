from xml.dom import INDEX_SIZE_ERR
import torch
import torch.nn as nn
from modules.bertembedding import BertEmbedding
from modules.instanceCL import InstanceCL
from modules.rnn import LSTM
from modules.crf import CRF
from modules.attn import Attention
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from data.snips.generate_slu_emb import slot2desp

        
def label_embedding_from_preLM(label_vocab, preLM_path):
    from transformers import BertTokenizer, BertModel
    def normalize(x, p=2, axis=-1):
        return x / np.linalg.norm(x, p, axis=axis, keepdims=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained(preLM_path)
    model = BertModel.from_pretrained(preLM_path)
    model = model.to(device)
    model.eval()
    label_embedding = []
    for lbl, _ in label_vocab:
        if lbl in slot2desp.keys():
            inputs = torch.tensor([tokenizer.encode(slot2desp[lbl])]).to(device)
        else:
            inputs = torch.tensor([tokenizer.encode(lbl)]).to(device)
        with torch.no_grad():
            last_hidden_states = model(inputs)[0]   # [1, seq_len, hidden_dim]
            seq_embedding = last_hidden_states.mean(dim=1)
        label_embedding.append(seq_embedding)
    label_embedding = torch.cat(label_embedding, dim=0).contiguous()
    label_embedding = label_embedding.cpu().numpy()
    label_embedding = normalize(label_embedding, p=2, axis=1)
    # np.savetxt('label.embedding', label_embedding)
    return label_embedding


class End2endSLUTagger(nn.Module):
    def __init__(self, vocabs, bert_embed_dim,
                 hidden_size,
                 num_type,
                 num_bound,
                 num_rnn_layer=1,
                 num_bert_layer=4,
                 dropout=0.0,
                 use_tr=False,
                 use_cl=False,
                 label_embed=None,
                 bert_model_path=None):
        super(End2endSLUTagger, self).__init__()
        self.vocabs = vocabs
        self.bert_embed_dim = bert_embed_dim
        self.num_type = num_type
        self.num_bound = num_bound
        self.dropout = dropout
        self.use_tr = use_tr
        self.use_cl = use_cl

        self.bert = BertEmbedding(bert_model_path,
                                #   num_bert_layer,
                                  merge='none',
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=False)

        hidden_size = self.bert_embed_dim // 2
        self.seq_encoder = LSTM(input_size=self.bert_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer,
                                dropout=dropout)

        self.bound_embedding = nn.Embedding(num_embeddings=num_bound,
                                            embedding_dim=10,
                                            padding_idx=0)
        nn.init.xavier_uniform_(self.bound_embedding.weight)

        self.hidden2boundary = nn.Linear(2*hidden_size, num_bound)

        self.tag_crf = CRF(num_tags=num_bound, batch_first=True)

        self.temperature = 1.0
        self.cl_temperature = 1.0
        self.hidden_proj = nn.Linear(2 * hidden_size + 10, 768)

        # self.slu_encoder = LSTM(input_size=2 * hidden_size + 10,
        #                         hidden_size=hidden_size,
        #                         num_layers=num_rnn_layer,
        #                         dropout=dropout)
        
        # TODO: self.vocabs['type']
        # self.label_proj = nn.Linear(768, 768, bias=False)
        self.label_embedding = nn.Parameter(torch.tensor(label_embedding_from_preLM(self.vocabs['slot'], bert_model_path)), requires_grad=True)

        if self.use_tr:
            self.template_enc = LSTM(self.bert_embed_dim,
                            hidden_size,
                            num_rnn_layer,
                            dropout=dropout)
            self.input_attn_layer = Attention(hidden_size*2)
            self.template_attn_layer = Attention(hidden_size*2)
        
        if self.use_cl:
            # self.cl_enc = LSTM(self.bert_embed_dim,
            #                 hidden_size,
            #                 num_rnn_layer,
            #                 dropout=dropout)
            # self.cl_enc = nn.Linear(self.bert_embed_dim,
            #                         hidden_size)
            # self.cl_loss = InstanceCL()
            pass
        
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
        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append((name, param))
        return other_params

    # def evaluate(self, bert_inp, mask=None):
    #     '''Decoding Rules:
    #     以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
    #     如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
    #     '''
    #     bert_repr = self.bert(*bert_inp)
    #     enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())
    #     boundary_score = self.hidden2boundary(enc_out)

    #     pred_bound = self.tag_decode(boundary_score, mask)

    #     bound_embed = self.bound_embedding(pred_bound)
    #     type_logits = self.hidden_proj(torch.cat((enc_out, bound_embed), dim=-1).contiguous())
    #     type_logits = F.normalize(type_logits, p=2, dim=-1)
    #     type_score = torch.matmul(type_logits, self.label_embedding.T) / self.temperature  # (B, N, D) x (D, K)
    #     pred_type = type_score.data.argmax(dim=-1)
    #     return pred_bound, pred_type

    # def forward(self, bert_inp, boundary_labels, type_labels, O_tag_idx, mask=None):
    #     ''' boundary tags -> type tags: B I E S  -> LOC PER ...
    #     :param bert_inps: bert_ids, segments, bert_masks, bert_lens
    #     :param mask: (bs, seq_len)  0 for padding
    #     :return:
    #     '''
    #     bert_repr = self.bert(*bert_inp)
    #     if self.training:
    #         bert_repr = timestep_dropout(bert_repr, p=self.dropout)

    #     enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())
    #     if self.training:
    #         enc_out = timestep_dropout(enc_out, p=self.dropout)

    #     boundary_score = self.hidden2boundary(enc_out)

    #     boundary_loss = self.tag_loss(boundary_score, boundary_labels, mask, reduction='none')
    
    #     # not differentiable
    #     # pred_bound = self.tag_decode(boundary_score, mask)
    #     # bound_embed = self.bound_embedding(pred_bound)
    #     # # bound_embed = self.bound_embedding(boundary_labels)
         
    #     # differentiable
    #     bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), self.bound_embedding.weight)
    #     # pred_bound = self.tag_decode(boundary_score, mask)

    #     type_logits = self.hidden_proj(torch.cat((enc_out, bound_embed), dim=-1).contiguous())
    #     type_logits = F.normalize(type_logits, p=2, dim=-1)
    #     type_score = torch.matmul(type_logits, self.label_embedding.T) / self.temperature  # (B, N, D) x (D, K)
    #     type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
    #     # if torch.rand(1).item() < 0.5:
    #     #     type_mask = (pred_bound.ne(O_tag_idx) * mask).float()
    #     # else:
    #     #     type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
    #     # type_loss = torch.sum(type_loss * type_mask, dim=1)  # (B, )

    #     # differentiable
    #     type_loss = torch.sum(type_loss, dim=1)
    #     loss = torch.mean(boundary_loss + type_loss)
    #     return loss, {"bio_loss": boundary_loss.mean().item(), "slu_loss": type_loss.mean().item()}, enc_out

    # TODO: KL loss
    def get_smooth_label(self, type_labels):
        onehot_labels = F.one_hot(type_labels, num_classes=len(self.vocabs["slot"]))
        bsz, seq_len, num_labels = onehot_labels.size()

        # proj_label_emb = self.proj_slot_emb(self.label_embedding)
        slots_similarity = (self.label_embedding @ self.label_embedding.T) # / (torch.linalg.norm(self.label_embedding, dim=-1)**2)
        slot_labels = onehot_labels + slots_similarity.index_select(dim=0, index=type_labels.view(-1)).view(bsz, seq_len, -1)
        return F.softmax(onehot_labels + slot_labels, dim=-1)


    def evaluate(self, bert_inp, mask=None):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        '''
        bert_repr = self.bert(*bert_inp)
        enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())
        boundary_score = self.hidden2boundary(enc_out)

        pred_bound = self.tag_decode(boundary_score, mask)

        bound_embed = self.bound_embedding(pred_bound)
        # bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), self.bound_embedding.weight)

        # shared lstm
        type_logits = self.hidden_proj(torch.cat((bert_repr, bound_embed), dim=-1).contiguous())
        # type_logits = self.hidden_proj(torch.cat((bert_repr, bound_embed), dim=-1).contiguous())
        type_logits = F.normalize(type_logits, p=2, dim=-1)
        type_score = torch.matmul(type_logits, self.label_embedding.T) / self.temperature  # (B, N, D) x (D, K)
        pred_type = type_score.data.argmax(dim=-1)
        return pred_bound, pred_type

    def forward(self, bert_inp, boundary_labels, type_labels, O_tag_idx, mask=None):
        ''' boundary tags -> type tags: B I E S  -> LOC PER ...
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        bert_repr = self.bert(*bert_inp)
        if self.training:
            bert_repr = F.dropout(bert_repr, p=self.dropout, training=self.training)

        enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())
        if self.training:
            enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)

        boundary_score = self.hidden2boundary(enc_out)

        boundary_loss = self.tag_loss(boundary_score, boundary_labels, mask, reduction='none')
    
        # not differentiable
        # pred_bound = self.tag_decode(boundary_score, mask)
        # bound_embed = self.bound_embedding(pred_bound)
        # # bound_embed = self.bound_embedding(boundary_labels)
         
        # differentiable
        bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), self.bound_embedding.weight)
        # pred_bound = self.tag_decode(boundary_score, mask)

        # shared lstm
        type_logits = self.hidden_proj(torch.cat((bert_repr, bound_embed), dim=-1).contiguous())
        # type_logits = self.hidden_proj(torch.cat((bert_repr, bound_embed), dim=-1).contiguous())
        type_logits = F.normalize(type_logits, p=2, dim=-1)
        type_score = torch.matmul(type_logits, self.label_embedding.T) / self.temperature  # (B, N, D) x (D, K)

        # smoothing loss
        # smooth_labels = self.get_smooth_label(type_labels)
        # type_loss = F.kl_div(type_score, smooth_labels, reduction='batchmean')
        
        # ignore index=-1, pad & O
        type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        # if torch.rand(1).item() < 0.5:
        #     type_mask = (pred_bound.ne(O_tag_idx) * mask).float()
        # else:
        #     type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        # type_loss = torch.sum(type_loss * type_mask, dim=1)  # (B, )

        # differentiable
        type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        type_loss = torch.sum(type_loss * type_mask, dim=1)
        loss = torch.mean(boundary_loss + type_loss)

        return loss, {"bio_loss": boundary_loss.mean().item(), "slu_loss": type_loss.mean().item()}, bert_repr
        # loss = torch.mean(boundary_loss) + type_loss
        # return loss, {"bio_loss": boundary_loss.mean().item(), "slu_loss": type_loss.item()}, enc_out


    
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
    
    def get_template_reprs(self, templates, token_mask, templates_mask, bert_repr):
        template_emb = self.bert(*templates)
        template_emb = F.dropout(template_emb, p=self.dropout, training=self.training)
        # if self.freeze_emb:
        #     template_embedding = self.embedding(templates).detach()
        #     template_embedding = F.dropout(template_embedding, p=self.dropout, training=self.training)
        # else:
        #     template_embedding = F.dropout(self.embedding(templates), p=self.dropout, training=self.training)
        template0_emb = template_emb[0::3]
        template1_emb = template_emb[1::3]
        template2_emb = template_emb[2::3]

        h_n1 = template0_emb[:, 0:1]
        h_n2 = template1_emb[:, 0:1]
        h_n3 = template2_emb[:, 0:1]
        
        # _, h_n1 = self.template_enc(template0_emb, non_pad_mask=templates_mask[0::3].cpu())
        # _, h_n2 = self.template_enc(template1_emb, non_pad_mask=templates_mask[1::3].cpu())
        # _, h_n3 = self.template_enc(template2_emb, non_pad_mask=templates_mask[2::3].cpu())
        # _, hn = self.template_enc(bert_repr, non_pad_mask=token_mask.cpu())
        # (bsz, 3, hidden_size)
        template_repr = torch.stack((h_n1, h_n2, h_n3), dim=1)

        return bert_repr[:, 0:1], template_repr
    
    def compute_contrastive_loss(self, templates, templates_mask, bert_reprs):
        template_reprs = self.bert(*templates)
        
        if self.training:
            template_reprs = F.dropout(template_reprs, p=self.dropout, training=self.training)
        # template_reprs = self.cl_enc(template_emb, non_pad_mask=templates_mask.cpu())[0]
        # if self.training:
        #     template_reprs = timestep_dropout(template_reprs, p=self.dropout)
        
        cl_loss = 0.
        query = F.normalize(bert_reprs, dim=-1)
        template_reprs = F.normalize(template_reprs, dim=-1)
        for i in range(template_reprs.size(0)//3):
            pos = torch.exp(F.cosine_similarity(query[i, 0:1], template_reprs[0::3][i, 0:1]) / self.cl_temperature)
            neg_1 = torch.exp(F.cosine_similarity(query[i, 0:1], template_reprs[1::3][i, 0:1]) / self.cl_temperature)
            neg_2 = torch.exp(F.cosine_similarity(query[i, 0:1], template_reprs[2::3][i, 0:1]) / self.cl_temperature)
            cl_loss -= torch.log(pos / (pos + neg_1 + neg_2))
        return cl_loss[0] / (template_reprs.size(0)//3)

    # def compute_contrastive_loss(self, templates, templates_mask, lstm_reprs):
    #     bert_ids, segments, bert_masks, bert_lens = templates

    #     bert_ids = torch.cat([bert_ids, bert_ids.clone()], dim=0)
    #     segments = torch.cat([segments, bert_ids.clone()], dim=0)
    #     bert_masks = torch.cat([bert_masks, bert_ids.clone()], dim=0)
    #     bert_lens = torch.cat([bert_lens, bert_ids.clone()], dim=0)
    #     template_emb = self.bert(bert_ids=bert_ids, segments=segments, bert_masks=bert_masks, bert_lens=bert_lens)

    #     idxs = torch.arange(0, bert_ids.size(0))
    #     idxs_1 = idxs.unsqueeze(0)
    #     idxs_2 = (idxs)
    