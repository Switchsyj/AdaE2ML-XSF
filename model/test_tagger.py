import torch
import torch.nn as nn
from modules.bertembedding import BertEmbedding
from modules.rnn import LSTM
from modules.crf import CRF

from modules.attn import Attention
import torch.nn.functional as F
import pickle as pkl
import os
from data.snips.generate_slu_emb import domain2slot
from modules.dropout import timestep_dropout
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from data.snips.generate_slu_emb import slot2desp


class BinaryModel(nn.Module):
    def __init__(self, embedding, 
            word_embed_dim, 
            hidden_size, 
            num_rnn_layer,
            num_tag, 
            bidirectional,
            freeze_emb,
            dropout=0.0):
        super(BinaryModel, self).__init__()
        self.embedding = embedding
        self.word_embed_dim = word_embed_dim
        self.dropout = dropout
        self.freeze_emb = freeze_emb
        self.num_tag = num_tag

        self.seq_encoder = LSTM(input_size=self.word_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer,
                                dropout=dropout,
                                bidirectional=bidirectional)

        self.hidden2tag = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, self.num_tag)
        self.crf = CRF(self.num_tag, batch_first=True)
       
    def forward(self, input, mask):
        '''
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param start_pos: (bs, seq_len)
        :param end_pos: (bs, seq_len)
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        if self.freeze_emb:
            embedding = self.embedding(input).detach()
        else:
            embedding = self.embedding(input)
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        
        lstm_repr = self.seq_encoder(embedding, non_pad_mask=mask.cpu())[0]
        tag_repr = self.hidden2tag(lstm_repr)
        
        return tag_repr, lstm_repr
    
    def compute_loss(self, tag_repr, gold_tags, mask):
        return self.crf(tag_repr, gold_tags, mask, reduction='mean').neg()
    
class SLUSeqTagger(nn.Module):
    def __init__(self, vocabs, slot_emb_file, hidden_size, num_layers, bidirectional):
        super(SLUSeqTagger, self).__init__()
        self.vocabs = vocabs
        self.slu_enc = LSTM(hidden_size*2 if bidirectional else hidden_size,
                            hidden_size,
                            num_layers=1,
                            dropout=0.,
                            bidirectional=bidirectional)
                            
        assert os.path.exists(slot_emb_file)
        with open(slot_emb_file, 'rb') as f:
            self.slot_emb = pkl.load(f)
    
    def forward(self, domain, lstm_repr, bio_gold=None, slu_gold=None):
        """
        return each sample repr in the batch.
        """
        gold_slotname_list = []
        feature_list = []
        for i in range(domain.size(0)):
            sent_label = bio_gold[i]
            sent_dm = domain[i]
            sent_domain_slot = domain2slot[self.vocabs["domain"].idx2inst(sent_dm.item())]
            # (seq_len, 1)
            b_index = sent_label == self.vocabs["binary"].inst2idx('B')
            i_index = sent_label == self.vocabs["binary"].inst2idx('I')
            
            nonzero_b = torch.nonzero(b_index)

            if nonzero_b.size(0) == 0:
                return 0
            
            prev_index = 0
            sent_slot_feature = []
            sent_domain_based_gold_label = []
            for j in range(nonzero_b.size(0)):
                if j == 0 and j != nonzero_b.size(0) - 1:
                    prev_index = nonzero_b[j]
                    continue

                cur_index = nonzero_b[j]
                if not (j == 0 and j != nonzero_b.size(0)):
                    nonzero_i = torch.nonzero(i_index[prev_index: cur_index])
                    nonzero_i = (prev_index + nonzero_i).squeeze(1)
                    slot_indices = torch.cat((prev_index, nonzero_i), dim=0)
                    slot_lstm_repr = lstm_repr[i][slot_indices.unsqueeze(0)]
                    slot_repr = torch.sum(self.slu_enc(slot_lstm_repr)[0], dim=1, keepdim=True)  # (1, 1, hidden_dim)
                    sent_slot_feature.append(slot_repr.squeeze(0))
                    
                    # get domain indexed slu name.
                    slu_name = self.vocabs["slot"].idx2inst(slu_gold[i][prev_index].item()).split('-')[1]
                    sent_domain_based_gold_label.append(sent_domain_slot.index(slu_name))
                if j == nonzero_b.size(0) - 1:
                    nonzero_i = torch.nonzero(i_index[cur_index:])
                    nonzero_i = (cur_index + nonzero_i).squeeze(1)
                    slot_indices = torch.cat((cur_index, nonzero_i), dim=0)
                    slot_lstm_repr = lstm_repr[i][slot_indices.unsqueeze(0)]
                    slot_repr = torch.sum(self.slu_enc(slot_lstm_repr)[0], dim=1, keepdim=True)  # (1, 1, hidden_dim)
                    sent_slot_feature.append(slot_repr.squeeze(0))

                    # get domain indexed slu name.
                    slu_name = self.vocabs["slot"].idx2inst(slu_gold[i][cur_index].item()).split('-')[1]
                    sent_domain_based_gold_label.append(sent_domain_slot.index(slu_name))
                else:
                    prev_index = cur_index
            
            sent_slot_feature = torch.cat(sent_slot_feature, dim=0)  # (sent_num_slots, hidden_size)
            feature_list.append(sent_slot_feature)
            gold_slotname_list.append(torch.tensor(sent_domain_based_gold_label, dtype=torch.long, device=slu_gold.device))
        
        pred_slotname_list = []
        for i in range(domain.size(0)):
            domain_embedding = torch.FloatTensor(self.slot_emb[self.vocabs["domain"].idx2inst(domain[i].item())]).to(domain.device)
            pred_score = torch.matmul(feature_list[i], domain_embedding.T)  # (sent_num_slots, num_slots)
            pred_slotname_list.append(pred_score)
     
        return pred_slotname_list, gold_slotname_list
        
    def evaluate(self, domains, lstm_reprs, bio_pred=None):
        """
        return each sample repr in the batch.
        """
        pred_label = bio_pred
        bsz = domains.size(0)
        slot_feature = []
        for i in range(bsz):
            sent_label = pred_label[i]
            sent_dm = domains[i]
            # (seq_len, 1)
            b_index = sent_label == self.vocabs["binary"].inst2idx('B')
            i_index = sent_label == self.vocabs["binary"].inst2idx('I')
            nonzero_b = torch.nonzero(b_index)
    
            if nonzero_b.size(0) == 0:
                slot_feature.append([])
                continue
            
            prev_index = 0
            sent_slot_feature = []
            for j in range(nonzero_b.size(0)):
                if j == 0 and j != nonzero_b.size(0) - 1:
                    prev_index = nonzero_b[j]
                    continue
                
                cur_index = nonzero_b[j]
                if not (j == 0 and j != nonzero_b.size(0)):
                    nonzero_i = torch.nonzero(i_index[prev_index: cur_index])
                    nonzero_i = (prev_index + nonzero_i).squeeze(1)
                    slot_indices = torch.cat((prev_index, nonzero_i), dim=0)
                    slot_lstm_repr = lstm_reprs[i, slot_indices]
                    slot_repr = torch.sum(self.slu_enc(slot_lstm_repr.unsqueeze(0))[0], dim=1, keepdim=True)  # (1, 1, hidden_dim)
                    sent_slot_feature.append(slot_repr.squeeze(0))   
                    
                if j == nonzero_b.size(0) - 1:
                    nonzero_i = torch.nonzero(i_index[cur_index:])
                    nonzero_i = (cur_index + nonzero_i).squeeze(1)
                    slot_indices = torch.cat((cur_index, nonzero_i), dim=0)
                    slot_lstm_repr = lstm_reprs[i, slot_indices]
                    slot_repr = torch.sum(self.slu_enc(slot_lstm_repr.unsqueeze(0))[0], dim=1, keepdim=True)  # (1, 1, hidden_dim)
                    sent_slot_feature.append(slot_repr.squeeze(0))
                else:
                    prev_index = cur_index
            slot_feature.append(torch.cat(sent_slot_feature, dim=0).to(domains.device))  # (sent_num_slot, hidden_dim)
            
        ## make predictions:
        pred_score = []
        for i in range(bsz):
            sent_feature = slot_feature[i]  # (sent_num_slots, hidden_size)
            if sent_feature == []:
                pred_score.append(None)
            else:
                domain_embedding = torch.FloatTensor(self.slot_emb[self.vocabs["domain"].idx2inst(domains[i].item())]).to(domains.device)
                pred_score.append(torch.matmul(sent_feature, domain_embedding.T))  # (sent_num_slots, num_slots)
            # assert pred_score[-1].size(0) == domain_based_gold_label[i].size(0)

        return pred_score
        

class TemplateEncoder(nn.Module):
    def __init__(self, embedding, word_embed_dim, hidden_size, num_layers, bidirectional, dropout, freeze_emb):
        super(TemplateEncoder, self).__init__()
        self.embedding = embedding
        self.freeze_emb = freeze_emb
        self.dropout = dropout
        self.wrd_embed_dim = word_embed_dim
        self.template_enc = LSTM(self.wrd_embed_dim,
                            hidden_size,
                            num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.input_attn_layer = Attention(hidden_size*2)
        self.template_attn_layer = Attention(hidden_size*2)
        
    def forward(self, templates, token_mask, templates_mask, lstm_repr):
        if self.freeze_emb:
            template_embedding = self.embedding(templates).detach()
            template_embedding = F.dropout(template_embedding, p=self.dropout, training=self.training)
        else:
            template_embedding = F.dropout(self.embedding(templates), p=self.dropout, training=self.training)
        template0_emb = template_embedding[0::3]
        template1_emb = template_embedding[1::3]
        template2_emb = template_embedding[2::3]
        
        template0_hidden = self.template_enc(template0_emb, non_pad_mask=templates_mask[0::3].cpu())[0]
        template1_hidden = self.template_enc(template1_emb, non_pad_mask=templates_mask[1::3].cpu())[0]
        template2_hidden = self.template_enc(template2_emb, non_pad_mask=templates_mask[2::3].cpu())[0]
        
        template0_hidden = self.template_attn_layer(template0_hidden, mask=templates_mask[0::3])
        template1_hidden = self.template_attn_layer(template1_hidden, mask=templates_mask[1::3])
        template2_hidden = self.template_attn_layer(template2_hidden, mask=templates_mask[2::3])
        
        # (bsz, 3, hidden_size)
        templates_repr = torch.stack((template0_hidden, template1_hidden, template2_hidden), dim=1)
        input_repr = self.input_attn_layer(lstm_repr, token_mask)
        return input_repr, templates_repr
