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
                 num_bound,
                 num_rnn_layer=1,
                 dropout=0.0,
                 use_cl=False,
                 cl_type='cosine',
                 bert_model_path=None,
                 cl_temperature=1.0,
                 alpha=1.0,
                 beta=1.0,
                 ):
        super(End2endSLUTagger, self).__init__()
        self.vocabs = vocabs
        self.bert_embed_dim = bert_embed_dim
        self.num_bound = num_bound
        self.dropout = dropout

        self.bert = BertEmbedding(model_path=bert_model_path,
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
        
        # TODO: ReLU, GELU
        self.label_adapter = nn.Sequential(nn.Linear(self.bert_embed_dim, self.bert_embed_dim//4, bias=False),
                                            getattr(nn, 'GELU')(),
                                            # nn.Dropout(p=0.3),
                                            nn.Linear(self.bert_embed_dim//4, self.bert_embed_dim, bias=False))
        self.alpha = alpha
        self.beta = beta

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

        label_embedding = self.label_adapter(bert_repr[:, 1: num_type+1]) + bert_repr[:, 1: num_type+1]     
        token_repr = bert_repr[:, num_type+1:]

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
    
    def forward(self, bert_inp, num_type, boundary_labels, 
                type_labels, O_tag_idx, mask=None, train='utter'):
        ''' boundary tags -> type tags: B I E S  -> LOC PER ...
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        
        bert_repr = self.bert(*bert_inp)

        label_embedding = self.label_adapter(bert_repr[:, 1: num_type+1]) + bert_repr[:, 1: num_type+1]   
        token_repr = bert_repr[:, num_type+1:]
        
        token_repr = F.dropout(token_repr, p=self.dropout, training=self.training)
        enc_out, _ = self.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)

        boundary_score = self.hidden2boundary(enc_out)
        boundary_loss = self.tag_loss(boundary_score, boundary_labels, mask, reduction='none')
        
        bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), self.bound_embedding.weight)
        type_logits = self.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
        
        if train == 'utter':
            type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2).detach()) / self.temperature  # (B, N, D) x (B, D, K)
            type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
        elif train == 'label':
            type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1).detach(), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
            type_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), type_labels, ignore_index=0, reduction='none')  # (B, L)
              
        type_mask = (boundary_labels.ne(O_tag_idx) * mask).float()
        type_loss = torch.sum(type_loss * type_mask, dim=1)
        
        loss = torch.mean(boundary_loss + self.alpha * type_loss)

        # use tokenCL
        if self.use_cl and train == 'utter':
            cl_loss = self.tokencl(token_repr, type_labels, mask)
            loss += (self.beta * cl_loss)
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
