from modules.coach_module import Lstm, CRF, Attention
from data.snips.generate_slu_emb import domain2slot
import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict


class BinarySLUTagger(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 freeze_emb, 
                 emb_file, 
                 num_layers, 
                 hidden_size, 
                 dropout, 
                 bidirectional, 
                 vocabs):
        super(BinarySLUTagger, self).__init__()
        
        self.lstm = Lstm(num_layers, emb_dim, hidden_size, dropout, bidirectional, freeze_emb, emb_file, use_bert=True)
        self.num_binslot = len(vocabs['binary'])
        self.hidden_dim = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(self.hidden_dim, self.num_binslot)
        self.crf_layer = CRF(self.num_binslot)
        
    def forward(self, X):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_binslot)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        lstm_hidden = self.lstm(X)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        return prediction, lstm_hidden
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data for i, length in enumerate(lengths)]

        return prediction
    
    def crf_loss(self, inputs, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        crf_loss = self.crf_layer.loss(inputs, y)

        return crf_loss


class ContrastivePredictor(nn.Module):
    def __init__(self, emb_dim, hidden_size, bidirectional, trs_hidden_size, trs_layers, slot_emb_file, vocabs, tgt_dm, slot_embs):
        super(ContrastivePredictor, self).__init__()
        self.input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.lstm_enc = nn.LSTM(self.input_dim, trs_hidden_size//2, num_layers=trs_layers, bidirectional=True, batch_first=True)
        
        self.emb_dim = emb_dim
        self.slot_embs = defaultdict(list)
        self.all_slots_embs = {}
        for dm, slots in domain2slot.items():
            for _, slot in enumerate(slots):
                self.slot_embs[dm].append(slot_embs[slot])
                self.all_slots_embs[slot] = slot_embs[slot]
            self.slot_embs[dm] = torch.stack(self.slot_embs[dm])
                
        self.vocabs = vocabs
        self.tgt_dm = tgt_dm              
        
        linear1 = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        activation = nn.Tanh()
        drop = nn.Dropout(0.2)
        linear2 = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        
        nn.init.eye_(linear1.weight)
        nn.init.constant_(linear1.bias, 0.0)
        nn.init.eye_(linear2.weight)
        nn.init.constant_(linear2.bias, 0.0)

        self.slot_name_projection_for_context = nn.Sequential(linear1, activation, drop, linear2)

        linear1 = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        activation = nn.Tanh()
        drop = nn.Dropout(0.2)
        linear2 = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        
        nn.init.eye_(linear1.weight)
        nn.init.constant_(linear1.bias, 0.0)
        nn.init.eye_(linear2.weight)
        nn.init.constant_(linear2.bias, 0.0)
        self.slot_name_projection_for_slot = nn.Sequential(linear1, activation, drop, linear2)

        linear1 = nn.Linear(hidden_size*2, hidden_size*2, bias=True)
        activation = nn.Tanh()
        linear2 = nn.Linear(hidden_size*2, hidden_size*2, bias=True)
        self.slot_projection = nn.Sequential(linear1, activation, linear2)

        linear1 = nn.Linear(hidden_size*2, hidden_size*2, bias=True)
        activation = nn.Tanh()
        linear2 = nn.Linear(hidden_size*2, hidden_size*2, bias=True)
        self.context_projection = nn.Sequential(linear1, activation, linear2)
        
        self.smooth_loss = nn.KLDivLoss(reduction='sum')
        self.xent_loss = nn.CrossEntropyLoss(reduction='sum')
        
    def getSmoothLabel(self, slot_name, domain, device):
        slot_id = domain2slot[domain].index(slot_name)
        slot_embedding_unenc = self.all_slots_embs[slot_name].to(device)
        
        slot_embedding = self.slot_name_projection_for_slot(slot_embedding_unenc)
        slot_context_embedding = self.slot_name_projection_for_context(slot_embedding_unenc)
        
        tgt_slot_embedding_unenc = self.slot_embs[self.tgt_dm]
        tgt_slot_embedding = self.slot_name_projection_for_slot(tgt_slot_embedding_unenc)
        tgt_slot_context_embedding = self.slot_name_projection_for_context(tgt_slot_embedding_unenc)
        
        slot_embedding_enc = torch.cat([slot_embedding, slot_context_embedding], dim=-1)    # (1, emb_dim*2)
        tgt_embedding_enc = torch.cat([tgt_slot_embedding, tgt_slot_context_embedding], dim=-1)     # (num_slot, emb_dim*2)
        
        similarity = torch.cosine_similarity(slot_embedding_enc.expand(tgt_embedding_enc.size()).contiguous().detach(), tgt_embedding_enc.detach(), dim=1)
        one_hot = F.one_hot(torch.tensor(domain2slot[domain].index(slot_name), dtype=torch.int64), len(domain2slot[domain])).to(device) * 0.6
        similarity = similarity / similarity.sum() * 0.4
        smooth_label = torch.cat([one_hot, similarity])
        return smooth_label
    
    def getContextFeats(self, slot_feats, hidden_i, seq_len, indices):
        slot_feats = slot_feats.detach()
        index = torch.arange(hidden_i.size(0)).to(hidden_i.device)
        mask = (index < indices[0]) | ((index > indices[-1]) & (index < seq_len))
        attn = torch.matmul(slot_feats, hidden_i.transpose(0, 1).contiguous()) * mask.unsqueeze(0)       # (1, seq_len)
        masked_attn = torch.exp(attn - torch.max(attn))
        masked_attn /= masked_attn.sum()
        
        # (1, seq_len) * (seq_len, hidden_size)
        return torch.matmul(attn, hidden_i.contiguous())
    
    def forward(self, domains, hidden_layers, length, binary_preditions=None, binary_golds=None, final_golds=None):
        """
        Inputs:
            domains: domain list for each sample (bsz,)
            hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
            binary_predictions: predictions made by our model (bsz, seq_len)
            binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
            final_golds: used only in the training mode (bsz, seq_len)
        Outputs:
            pred_slotname_list: list of predicted slot names
            gold_slotname_list: list of gold slot names  (only return this in the training mode)
        """
        binary_labels = binary_golds if binary_golds is not None else binary_preditions

        feature_list = []
        context_list = []
        slot_name_list = []
        if final_golds is not None:
            # only in the training mode
            gold_slotname_list = []

        bsz = domains.size()[0]
        
        ### collect features of slot and their corresponding labels (gold_slotname) in this batch
        for i in range(bsz):
            dm_id = domains[i].item()
            domain_name = self.vocabs['domain'].idx2inst(dm_id)
            slot_list_based_domain = domain2slot[domain_name]  # a list of slot names

            # we can also add domain embeddings after transformer encoder
            hidden_i = hidden_layers[i]    # (seq_len, hidden_dim)

            ## collect range of slot name and hidden layers
            feature_each_sample = []
            context_feature_each_sample = []
            slot_name_each_sample = []
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
                gold_slotname_each_sample = []
            
            bin_label = binary_labels[i]
            # get indices of B and I
            B_list = bin_label == self.vocabs['binary'].inst2idx('B')
            I_list = bin_label == self.vocabs['binary'].inst2idx('I')
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]
            
            if num_slotname == 0:
                feature_list.append(feature_each_sample)
                context_list.append(context_feature_each_sample)
                continue

            for j in range(num_slotname):
                if j == 0 and j < num_slotname-1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_slotname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])

                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        indices = curr_index
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                    slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)

                    feature_each_sample.append(slot_feats.squeeze(0))
                    context_feats = self.getContextFeats(slot_feats, hidden_i, length[i], indices)
                    context_feature_each_sample.append(context_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = self.vocabs['slot'].idx2inst(final_gold_each_sample[prev_index].item()).split("-")[1]
                        slot_name_each_sample.append(slot_name)
                        if domain_name != self.tgt_dm:
                            gold_slotname_each_sample.append(self.getSmoothLabel(slot_name, domain_name, hidden_i.device))
                        else:
                            gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                
                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        indices = curr_index
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                
                    slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                    slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                   
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    feature_each_sample.append(slot_feats.squeeze(0))
                    context_feats = self.getContextFeats(slot_feats, hidden_i, length[i], indices)
                    context_feature_each_sample.append(context_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = self.vocabs['slot'].idx2inst(final_gold_each_sample[curr_index].item()).split("-")[1]
                        slot_name_each_sample.append(slot_name)
                        if domain_name != self.tgt_dm:
                            gold_slotname_each_sample.append(self.getSmoothLabel(slot_name, domain_name, hidden_i.device))
                        else:
                            gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                else:
                    prev_index = curr_index
            
            feature_each_sample = torch.stack(feature_each_sample)  # (num_slotname, hidden_dim)
            context_feature_each_sample = torch.stack(context_feature_each_sample)
            feature_list.append(feature_each_sample)
            context_list.append(context_feature_each_sample)
            slot_name_list.append(slot_name_each_sample)
            if final_golds is not None:
                if domain_name != self.tgt_dm:
                    gold_slotname_each_sample = torch.stack(gold_slotname_each_sample)
                else:
                    gold_slotname_each_sample = torch.LongTensor(gold_slotname_each_sample)   # (num_slotname)
                gold_slotname_list.append(gold_slotname_each_sample)

        ### predict slot names
        pred_slotname_list = []
        assert sum([len(x) != len(y) for x, y in zip(feature_list, context_list)]) == 0
        for i in range(bsz):
            dm_id = domains[i].item()
            domain_name = self.vocabs['domain'].idx2inst(dm_id)

            feature_each_sample = feature_list[i]       # (num_slotname, hidden_dim)  hidden_dim == emb_dim
            context_feature_each_sample = context_list[i]       # (num_slotname, hidden_dim)
            if len(feature_each_sample) == 0:
                # only in the evaluation phrase
                pred_slotname_each_sample = None
            else:
                if domain_name != self.tgt_dm:
                    src = self.slot_embs[domain_name]   # (slot_num, emb_dim)
                    tgt = self.slot_embs[self.tgt_dm]   # (slot_num, emb_dim)
                    slot_embs_based_domain_src = torch.cat([src, tgt], dim=0)
                else:
                    slot_embs_based_domain_src = self.slot_embs[self.tgt_dm]   # (slot_num, emb_dim)
                
                slot_embedding = self.slot_name_projection_for_slot(slot_embs_based_domain_src).transpose(0, 1)
                slot_feats = self.slot_projection(feature_each_sample)
                
                slot_context_embedding = self.slot_name_projection_for_context(slot_embs_based_domain_src).transpose(0, 1)
                context_feats = self.context_projection(context_feature_each_sample)
                
                pred_slotname_each_sample = torch.matmul(slot_feats, slot_embedding) # (num_slotname, slot_num)
                pred_slotname_each_sample_context = torch.matmul(context_feats, slot_context_embedding)
                pred_slotname_each_sample = pred_slotname_each_sample + 0.1 * pred_slotname_each_sample_context     # theta parameter
            
            pred_slotname_list.append(pred_slotname_each_sample)
        
        if final_golds is not None:
            # smoothing loss
            slot_num = 0.
            type_loss = 0.
            for pred, gold, dm_id in zip(pred_slotname_list, gold_slotname_list, domains):
                domain_name = self.vocabs['domain'].idx2inst(dm_id.item())
                slot_num += gold.size(0)
                if domain_name != self.tgt_dm:
                    type_loss += self.smooth_loss(F.log_softmax(pred, dim=-1), gold)
                    # type_loss += F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(gold, dim=-1), reduction='sum')
                else:
                    type_loss += F.cross_entropy(pred, gold, reduction='sum')
            type_loss /=  slot_num
               
            # slot contrastive loss
            slot_num = 0.
            con_loss = 0.
            for i in range(bsz):
                feature_each_sample = feature_list[i]
                slot_name_each_sample = slot_name_list[i]
                domain_name = self.vocabs['domain'].idx2inst(domains[i].item())
                assert len(feature_each_sample) == len(slot_name_each_sample)
                if len(feature_each_sample) == 0:
                    continue
                for feat, slot_name in zip(feature_each_sample, slot_name_each_sample):
                    slot_num += 1.
                    slot_embedding = []
                    slot_id = domain2slot[domain_name].index(slot_name)
                    slot_embedding.append(self.slot_embs[domain_name][slot_id])
                    slot_embedding.extend([v for k, v in self.all_slots_embs.items() if k != slot_name])
                    slot_embedding = self.slot_name_projection_for_slot(torch.stack(slot_embedding))
                    
                    feat = self.slot_projection(feat)
                    con_loss += self.contrastive_loss(feat, slot_embedding, 0.9)
            con_loss /= slot_num
            
            con_context_loss = 0.
            slot_num = 0.
            for i in range(bsz):
                feature_each_sample = feature_list[i]
                slot_name_each_sample = slot_name_list[i]
                domain_name = self.vocabs['domain'].idx2inst(domains[i].item())
                assert len(feature_each_sample) == len(slot_name_each_sample)
                if len(feature_each_sample) == 0:
                    continue
                for feat, slot_name in zip(feature_each_sample, slot_name_each_sample):
                    slot_num += 1.
                    slot_embedding = []
                    slot_id = domain2slot[domain_name].index(slot_name)
                    slot_embedding.append(self.slot_embs[domain_name][slot_id])
                    slot_embedding.extend([v for k, v in self.all_slots_embs.items() if k != slot_name])
                    slot_embedding = self.slot_name_projection_for_context(torch.stack(slot_embedding))
                    
                    feat = self.context_projection(feat)
                    con_context_loss += self.contrastive_loss(feat, slot_embedding, 0.9)
            con_context_loss /= slot_num
                    
        if final_golds is not None:
            # only in the training mode
            return type_loss, con_loss, con_context_loss
        else:
            return pred_slotname_list
        
    def contrastive_loss(self, query, keys, temperature=0.1):
        """
        Given the query vector and the keys matrix (the first vector is a positive sample by default, and the rest are negative samples), calculate the contrast learning loss, follow SimCLR
        :param query: shape=(d,)
        :param keys: shape=(39, d)
        :return: scalar
        """
        query = torch.nn.functional.normalize(query, dim=0)
        keys = torch.nn.functional.normalize(keys, dim=1)
        output = torch.nn.functional.cosine_similarity(query.unsqueeze(0), keys)  # (39,)
        numerator = torch.exp(output[0] / temperature)
        denominator = torch.sum(torch.exp(output / temperature))
        return -torch.log(numerator / denominator)
    
class SentRepreGenerator(nn.Module):
    def __init__(self, emb_dim, freeze_emb, emb_file, hidden_size, num_layers, dropout, bidirectional, vocabs):
        super(SentRepreGenerator, self).__init__()
        self.hidden_size = hidden_size * 2 if bidirectional else hidden_size
        
        # LSTM Encoder for template
        self.template_encoder = Lstm(num_layers, emb_dim, hidden_size, dropout, bidirectional, freeze_emb, emb_file, use_bert=True)

        # attention layers for templates and input sequences
        self.input_atten_layer = Attention(attention_size=self.hidden_size)
        self.template_attn_layer = Attention(attention_size=self.hidden_size)

    def forward(self, templates, tem_lengths, hidden_layers, x_lengths):
        """
        Inputs:
            templates: (bsz, 3, max_template_length)
            tem_lengths: (bsz,)
            hidden_layers: (bsz, max_length, hidden_size)
            x_lengths: (bsz,)
        Outputs:
            template_sent_repre: (bsz, 3, hidden_size)
            input_sent_repre: (bsz, hidden_size)
        """
        # generate templates sentence representation
        template0_hiddens = self.template_encoder(templates[0::3, :])
        template1_hiddens = self.template_encoder(templates[1::3, :])
        template2_hiddens = self.template_encoder(templates[2::3, :])

        template0_repre, _ = self.template_attn_layer(template0_hiddens, tem_lengths[0::3])
        template1_repre, _ = self.template_attn_layer(template1_hiddens, tem_lengths[1::3])
        template2_repre, _ = self.template_attn_layer(template2_hiddens, tem_lengths[2::3])

        templates_repre = torch.stack((template0_repre, template1_repre, template2_repre), dim=1)  # (bsz, 3, hidden_size)

        # generate input sentence representations
        input_repre, _ = self.input_atten_layer(hidden_layers, x_lengths)

        return templates_repre, input_repre
    
class BertContrastivePredictor(nn.Module):
    def __init__(self, emb_dim, hidden_size, bidirectional, trs_hidden_size, trs_layers, slot_emb_file, vocabs, tgt_dm, theta, smooth, temp):
        super(BertContrastivePredictor, self).__init__()
        self.input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.lstm_enc = nn.LSTM(self.input_dim, trs_hidden_size//2, num_layers=trs_layers, bidirectional=True, batch_first=True)
        
        self.emb_dim = emb_dim
                
        self.vocabs = vocabs
        self.tgt_dm = tgt_dm    
        self.theta = theta        
        self.smooth = smooth
        self.temp = temp  
        
        linear1 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        activation = nn.Tanh()
        drop = nn.Dropout(0.2)
        linear2 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        
        nn.init.eye_(linear1.weight)
        nn.init.constant_(linear1.bias, 0.0)
        nn.init.eye_(linear2.weight)
        nn.init.constant_(linear2.bias, 0.0)

        self.slot_name_projection_for_context = nn.Sequential(linear1, activation, drop, linear2)

        linear1 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        activation = nn.Tanh()
        drop = nn.Dropout(0.2)
        linear2 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        
        nn.init.eye_(linear1.weight)
        nn.init.constant_(linear1.bias, 0.0)
        nn.init.eye_(linear2.weight)
        nn.init.constant_(linear2.bias, 0.0)
        self.slot_name_projection_for_slot = nn.Sequential(linear1, activation, drop, linear2)

        linear1 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        activation = nn.Tanh()
        linear2 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.slot_projection = nn.Sequential(linear1, activation, linear2)

        linear1 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        activation = nn.Tanh()
        linear2 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.context_projection = nn.Sequential(linear1, activation, linear2)
        
        self.smooth_loss = nn.KLDivLoss(reduction='sum')
        self.xent_loss = nn.CrossEntropyLoss(reduction='sum')
    
    def getDomainEmbedding(self, dm, all_slots_embs):
        dm_emb = []
        for slot in domain2slot[dm]:
            dm_emb.append(all_slots_embs[slot])
        return torch.stack(dm_emb)
    
    def getSmoothLabel(self, slot_name, domain, slots_embs_based_on_domain, device):
        slot_id = domain2slot[domain].index(slot_name)
        slot_embedding_unenc = slots_embs_based_on_domain[domain][slot_id]
        
        slot_embedding = self.slot_name_projection_for_slot(slot_embedding_unenc)
        slot_context_embedding = self.slot_name_projection_for_context(slot_embedding_unenc)
        
        tgt_slot_embedding_unenc = slots_embs_based_on_domain[self.tgt_dm]
        tgt_slot_embedding = self.slot_name_projection_for_slot(tgt_slot_embedding_unenc)
        tgt_slot_context_embedding = self.slot_name_projection_for_context(tgt_slot_embedding_unenc)
        
        slot_embedding_enc = torch.cat([slot_embedding, slot_context_embedding], dim=-1)    # (1, emb_dim*2)
        tgt_embedding_enc = torch.cat([tgt_slot_embedding, tgt_slot_context_embedding], dim=-1)     # (num_slot, emb_dim*2)
        
        slot_embedding_enc = slot_embedding_enc.expand(tgt_embedding_enc.size()).contiguous().detach()
        tgt_embedding_enc = tgt_embedding_enc.detach()
        
        similarity = torch.cosine_similarity(slot_embedding_enc, tgt_embedding_enc, dim=1)
        one_hot = F.one_hot(torch.tensor(domain2slot[domain].index(slot_name), dtype=torch.int64), len(domain2slot[domain])).to(device) * self.smooth
        similarity = similarity / similarity.sum() * (1.0 - self.smooth)
        smooth_label = torch.cat([one_hot, similarity])
        return smooth_label
    
    def getContextFeats(self, slot_feats, hidden_i, seq_len, indices):
        slot_feats = slot_feats.detach()
        index = torch.arange(hidden_i.size(0)).to(hidden_i.device)
        mask = (index < indices[0]) | ((index > indices[-1]) & (index < seq_len))
        attn = torch.matmul(slot_feats, hidden_i.transpose(0, 1).contiguous()) * mask.unsqueeze(0)       # (1, seq_len)
        masked_attn = torch.exp(attn - torch.max(attn))
        masked_attn /= masked_attn.sum()
        
        # (1, seq_len) * (seq_len, hidden_size)
        return torch.matmul(attn, hidden_i.contiguous())
    
    def forward(self, domains, hidden_layers, length, slots_embs_based_on_domain, binary_preditions=None, binary_golds=None, final_golds=None):
        """
        Inputs:
            domains: domain list for each sample (bsz,)
            hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
            binary_predictions: predictions made by our model (bsz, seq_len)
            binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
            final_golds: used only in the training mode (bsz, seq_len)
        Outputs:
            pred_slotname_list: list of predicted slot names
            gold_slotname_list: list of gold slot names  (only return this in the training mode)
        """
        binary_labels = binary_golds if binary_golds is not None else binary_preditions

        feature_list = []
        context_list = []
        slot_name_list = []
        if final_golds is not None:
            # only in the training mode
            gold_slotname_list = []

        bsz = domains.size()[0]
        
        ### collect features of slot and their corresponding labels (gold_slotname) in this batch
        for i in range(bsz):
            dm_id = domains[i].item()
            domain_name = self.vocabs['domain'].idx2inst(dm_id)
            slot_list_based_domain = domain2slot[domain_name]  # a list of slot names

            # we can also add domain embeddings after transformer encoder
            hidden_i = hidden_layers[i]    # (seq_len, hidden_dim)

            ## collect range of slot name and hidden layers
            feature_each_sample = []
            context_feature_each_sample = []
            slot_name_each_sample = []
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
                gold_slotname_each_sample = []
            
            bin_label = binary_labels[i]
            # get indices of B and I
            B_list = bin_label == self.vocabs['binary'].inst2idx('B')
            I_list = bin_label == self.vocabs['binary'].inst2idx('I')
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]
            
            if num_slotname == 0:
                feature_list.append(feature_each_sample)
                context_list.append(context_feature_each_sample)
                continue

            for j in range(num_slotname):
                if j == 0 and j < num_slotname-1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_slotname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])

                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        indices = curr_index
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                    slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)

                    feature_each_sample.append(slot_feats.squeeze(0))
                    context_feats = self.getContextFeats(slot_feats, hidden_i, length[i], indices)
                    context_feature_each_sample.append(context_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = self.vocabs['slot'].idx2inst(final_gold_each_sample[prev_index].item()).split("-")[1]
                        slot_name_each_sample.append(slot_name)
                        if domain_name != self.tgt_dm:
                            gold_slotname_each_sample.append(self.getSmoothLabel(slot_name, domain_name, slots_embs_based_on_domain, hidden_i.device))
                        else:
                            gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                
                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        indices = curr_index
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                
                    slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                    slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                   
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    feature_each_sample.append(slot_feats.squeeze(0))
                    context_feats = self.getContextFeats(slot_feats, hidden_i, length[i], indices)
                    context_feature_each_sample.append(context_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = self.vocabs['slot'].idx2inst(final_gold_each_sample[curr_index].item()).split("-")[1]
                        slot_name_each_sample.append(slot_name)
                        if domain_name != self.tgt_dm:
                            gold_slotname_each_sample.append(self.getSmoothLabel(slot_name, domain_name, slots_embs_based_on_domain, hidden_i.device))
                        else:
                            gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                else:
                    prev_index = curr_index
            
            feature_each_sample = torch.stack(feature_each_sample)  # (num_slotname, hidden_dim)
            context_feature_each_sample = torch.stack(context_feature_each_sample)
            feature_list.append(feature_each_sample)
            context_list.append(context_feature_each_sample)
            slot_name_list.append(slot_name_each_sample)
            if final_golds is not None:
                if domain_name != self.tgt_dm:
                    gold_slotname_each_sample = torch.stack(gold_slotname_each_sample)
                else:
                    gold_slotname_each_sample = torch.LongTensor(gold_slotname_each_sample)   # (num_slotname)
                gold_slotname_list.append(gold_slotname_each_sample)

        ### predict slot names
        pred_slotname_list = []
        assert sum([len(x) != len(y) for x, y in zip(feature_list, context_list)]) == 0
        for i in range(bsz):
            dm_id = domains[i].item()
            domain_name = self.vocabs['domain'].idx2inst(dm_id)

            feature_each_sample = feature_list[i]       # (num_slotname, hidden_dim)  hidden_dim == emb_dim
            context_feature_each_sample = context_list[i]       # (num_slotname, hidden_dim)
            if len(feature_each_sample) == 0:
                # only in the evaluation phrase
                pred_slotname_each_sample = None
            else:
                if domain_name != self.tgt_dm:
                    src = slots_embs_based_on_domain[domain_name]   # (slot_num, emb_dim)
                    tgt = slots_embs_based_on_domain[self.tgt_dm]   # (slot_num, emb_dim)
                    slot_embs_based_domain_src = torch.cat([src, tgt], dim=0)
                else:
                    slot_embs_based_domain_src = slots_embs_based_on_domain[self.tgt_dm]   # (slot_num, emb_dim)
                
                slot_embedding = self.slot_name_projection_for_slot(slot_embs_based_domain_src).transpose(0, 1)
                slot_feats = self.slot_projection(feature_each_sample)
                
                slot_context_embedding = self.slot_name_projection_for_context(slot_embs_based_domain_src).transpose(0, 1)
                context_feats = self.context_projection(context_feature_each_sample)
                
                pred_slotname_each_sample = torch.matmul(slot_feats, slot_embedding) # (num_slotname, slot_num)
                pred_slotname_each_sample_context = torch.matmul(context_feats, slot_context_embedding)
                pred_slotname_each_sample = pred_slotname_each_sample + self.theta * pred_slotname_each_sample_context     # theta parameter
            
            pred_slotname_list.append(pred_slotname_each_sample)
        
        if final_golds is not None:
            # smoothing loss
            slot_num = 0.
            type_loss = 0.
            for pred, gold, dm_id in zip(pred_slotname_list, gold_slotname_list, domains):
                domain_name = self.vocabs['domain'].idx2inst(dm_id.item())
                slot_num += gold.size(0)
                if domain_name != self.tgt_dm:
                    type_loss += self.smooth_loss(F.log_softmax(pred, dim=-1), gold)
                else:
                    type_loss += F.cross_entropy(pred, gold, reduction='sum')
            type_loss /=  slot_num
               
            # slot contrastive loss
            slot_num = 0.
            con_loss = 0.
            
            all_slot_embedding = defaultdict(list)
            for dm, embs in slots_embs_based_on_domain.items():
                for i, emb in enumerate(embs): 
                    all_slot_embedding[domain2slot[dm][i]].append(emb)
            for k, v in all_slot_embedding.items():
                all_slot_embedding[k] = torch.stack(v).mean(dim=0)
            
            for i in range(bsz):
                feature_each_sample = feature_list[i]
                slot_name_each_sample = slot_name_list[i]
                domain_name = self.vocabs['domain'].idx2inst(domains[i].item())
                assert len(feature_each_sample) == len(slot_name_each_sample)
                if len(feature_each_sample) == 0:
                    continue
                for feat, slot_name in zip(feature_each_sample, slot_name_each_sample):
                    slot_num += 1.
                    slot_embedding = []
                    slot_id = domain2slot[domain_name].index(slot_name)
                    slot_embedding.append(all_slot_embedding[slot_name])
                    slot_embedding.extend([v for k, v in all_slot_embedding.items() if k != slot_name])
                    slot_embedding = self.slot_name_projection_for_slot(torch.stack(slot_embedding))
                    
                    feat = self.slot_projection(feat)
                    con_loss += self.contrastive_loss(feat, slot_embedding, self.temp)
            con_loss /= slot_num
            
            con_context_loss = 0.
            slot_num = 0.
            for i in range(bsz):
                feature_each_sample = feature_list[i]
                slot_name_each_sample = slot_name_list[i]
                domain_name = self.vocabs['domain'].idx2inst(domains[i].item())
                assert len(feature_each_sample) == len(slot_name_each_sample)
                if len(feature_each_sample) == 0:
                    continue
                for feat, slot_name in zip(feature_each_sample, slot_name_each_sample):
                    slot_num += 1.
                    slot_embedding = []
                    slot_id = domain2slot[domain_name].index(slot_name)
                    slot_embedding.append(all_slot_embedding[slot_name])
                    slot_embedding.extend([v for k, v in all_slot_embedding.items() if k != slot_name])
                    slot_embedding = self.slot_name_projection_for_context(torch.stack(slot_embedding))
                    
                    feat = self.context_projection(feat)
                    con_context_loss += self.contrastive_loss(feat, slot_embedding, self.temp)
            con_context_loss /= slot_num
                    
        if final_golds is not None:
            # only in the training mode
            return type_loss, con_loss, con_context_loss
        else:
            return pred_slotname_list
        
    def contrastive_loss(self, query, keys, temperature=0.1):
        """
        Given the query vector and the keys matrix (the first vector is a positive sample by default, and the rest are negative samples), calculate the contrast learning loss, follow SimCLR
        :param query: shape=(d,)
        :param keys: shape=(39, d)
        :return: scalar
        """
        query = torch.nn.functional.normalize(query, dim=0)
        keys = torch.nn.functional.normalize(keys, dim=1)
        output = torch.nn.functional.cosine_similarity(query.unsqueeze(0), keys)  # (39,)
        numerator = torch.exp(output[0] / temperature)
        denominator = torch.sum(torch.exp(output / temperature))
        return -torch.log(numerator / denominator)