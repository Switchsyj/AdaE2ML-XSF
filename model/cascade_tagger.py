import torch
import torch.nn as nn
from modules.rnn import LSTM
from modules.dropout import *
from modules.crf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from modules.bertembedding import BertEmbedding
import numpy as np

slot2desp = {'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 'facility': 'facility', 'movie_name': 'movie name', 'location_name': 'location name', 'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 'poi': 'position', 'party_size_description': 'person', 'served_dish': 'served dish', 'genre': 'genre', 'current_location': 'current location', 'object_select': 'this current', 'album': 'album', 'object_name': 'object name', 'state': 'location', 'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 'entity_name': 'entity name', 'object_type': 'object type', 'playlist_owner': 'owner', 'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 'best_rating': 'best rating', 'rating_unit': 'rating unit', 'year': 'year', 'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}


def label_embedding_from_preLM(label_vocab, preLM_path):
    from transformers import BertTokenizer, BertModel
    def normalize(x, p=2, axis=-1):
        return x / np.linalg.norm(x, p, axis=axis, keepdims=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class CascadeTagger(nn.Module):
    def __init__(self, vocabs, bert_embed_dim,
                 hidden_size,
                 num_type,
                 num_bound,
                 num_rnn_layer=1,
                 num_bert_layer=4,
                 dropout=0.0,
                 use_crf=True,
                 label_embed=None,
                 bert_model_path=None):
        super(CascadeTagger, self).__init__()
        self.vocabs = vocabs
        self.bert_embed_dim = bert_embed_dim
        self.num_type = num_type
        self.num_bound = num_bound
        self.dropout = dropout
        self.use_crf = use_crf

        self.bert = BertEmbedding(bert_model_path,
                                  num_bert_layer,
                                  merge='linear',
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

        if self.use_crf:
            self.tag_crf = CRF(num_tags=num_bound, batch_first=True)

        self.temperature = 1.
        self.hidden_proj = nn.Linear(2 * hidden_size + 10, 768)
        
        # TODO: self.vocabs['type']
        self.label_embedding = nn.Parameter(torch.tensor(label_embedding_from_preLM(self.vocabs['slot'], bert_model_path)), requires_grad=False)

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

    def inference(self, bert_inp, mask=None):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        '''
        bert_repr = self.bert(*bert_inp)
        enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())
        boundary_score = self.hidden2boundary(enc_out)
        if self.use_crf:
            pred_bound = self.tag_decode(boundary_score, mask)
        else:
            pred_bound = boundary_score.data.argmax(dim=-1) * mask.long()

        bound_embed = self.bound_embedding(pred_bound)
        type_logits = self.hidden_proj(torch.cat((enc_out, bound_embed), dim=-1).contiguous())
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
            bert_repr = timestep_dropout(bert_repr, p=self.dropout)

        enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())
        if self.training:
            enc_out = timestep_dropout(enc_out, p=self.dropout)

        boundary_score = self.hidden2boundary(enc_out)
        if self.use_crf:
            boundary_loss = self.tag_loss(boundary_score, boundary_labels, mask, reduction='none')
            pred_bound = self.tag_decode(boundary_score, mask)
        else:
            boundary_loss = F.cross_entropy(boundary_score.transpose(1, 2).contiguous(), boundary_labels, ignore_index=0, reduction='none') * mask.float()
            weights_of_loss = boundary_labels.ne(O_tag_idx).float() + 0.5
            boundary_loss = (boundary_loss * weights_of_loss).sum(dim=1)  # (B, )
            pred_bound = boundary_score.data.argmax(dim=-1)

        bound_embed = self.bound_embedding(pred_bound)
        type_logits = self.hidden_proj(torch.cat((enc_out, bound_embed), dim=-1).contiguous())
        type_logits = F.normalize(type_logits, p=2, dim=-1)
        type_score = torch.matmul(type_logits, self.label_embedding.T) / self.temperature  # (B, N, D) x (D, K)
        type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        if torch.rand(1).item() < 0.7:
            type_mask = (pred_bound.ne(O_tag_idx) * mask).float()
        else:
            type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        type_loss = torch.sum(type_loss * type_mask, dim=1)  # (B, )
        loss = torch.mean(boundary_loss + type_loss)
        return loss

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