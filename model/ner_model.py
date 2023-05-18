from audioop import bias
import torch
import torch.nn as nn
from modules.bertembedding import BertEmbedding
from modules.tokenCL import TokenCL
from modules.rnn import LSTM
from modules.crf import CRF
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class End2endSLUTagger(nn.Module):
    def __init__(self, vocabs, bert_embed_dim,
                 num_label,
                 num_rnn_layer=1,
                 num_bert_layer=4,
                 dropout=0.0,
                 bert_model_path=None,
                 ):
        super(End2endSLUTagger, self).__init__()
        self.vocabs = vocabs
        self.bert_embed_dim = bert_embed_dim
        self.num_label = num_label
        self.dropout = dropout

        self.bert = BertEmbedding(model_path=bert_model_path,
                                  merge='none',
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=False)

        hidden_size = self.bert_embed_dim // 2
        self.seq_encoder = LSTM(input_size=self.bert_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer)

        self.hidden2boundary = nn.Linear(2*hidden_size, num_label)
        self.tag_crf = CRF(num_tags=num_label, batch_first=True)

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

    def evaluate(self, bert_inp, mask=None):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        '''
        token_repr = self.bert(*bert_inp)[:, 1:]

        # enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        enc_out = token_repr
        
        score = self.hidden2boundary(enc_out)

        pred_label = self.tag_decode(score, mask)
        return pred_label

    def forward(self, bert_inp, labels, mask=None):
        ''' boundary tags -> type tags: B I E S  -> LOC PER ...
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        
        token_repr = self.bert(*bert_inp)[:, 1:]
        
        token_repr = F.dropout(token_repr, p=self.dropout, training=self.training)
        # enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        # enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)
        enc_out = token_repr

        score = self.hidden2boundary(enc_out)
        tag_loss = self.tag_loss(score, labels, mask, reduction='none')
            
        return tag_loss.mean(), {"tag_loss": tag_loss.mean().item()}

    
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
