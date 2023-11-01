import os
import torch
import torch.nn as nn
from transformers import BertModel


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers=1, fine_tune=True, use_proj=False, proj_dim=256):
        super(BertEmbedding, self).__init__()
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.fine_tune = fine_tune
        # Pretrained Bert model.
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        
        self.bert_layers = self.bert.config.num_hidden_layers + 1  # including embedding layer
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size

        if not self.fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        if self.use_proj:
            self.proj = nn.Linear(self.hidden_size, self.proj_dim)
            self.hidden_size = self.proj_dim
        else:
            self.proj = None

    def save_bert(self, save_dir):
        # saved into config file and model
        assert os.path.isdir(save_dir)
        self.bert.save_pretrained(save_dir)
        print('BERT Saved !!!')

    def forward(self, bert_ids, segments, bert_mask, bert_lens):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param segments: (bz, bpe_seq_len)  只有一个句子，全0
        :param bert_mask: (bz, bpe_seq_len)  经过bpe切词
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        bert_mask = bert_mask.type_as(mask)

        if self.fine_tune:
            last_enc_out, _, _ = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)
        else:
            with torch.no_grad():
                last_enc_out, _, _ = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)

        enc_out = last_enc_out

        # 根据bert piece长度切分
        bert_chunks = enc_out[bert_mask].split(bert_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(bz, seq_len, self.bert.config.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        output = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
       
        if self.proj:
            return self.proj(output)
        else:
            return output
