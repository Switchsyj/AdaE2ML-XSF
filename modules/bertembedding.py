import os
import torch
import torch.nn as nn
from transformers import BertModel
from modules.scalemix import ScalarMix
import torch.nn.functional as F


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers=1, merge='none', fine_tune=True, use_proj=False, proj_dim=256):
        super(BertEmbedding, self).__init__()
        assert merge in ['none', 'linear', 'mean', 'attn']
        self.merge = merge
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.fine_tune = fine_tune
        # Pretrained Bert model.
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        
        self.bert_layers = self.bert.config.num_hidden_layers + 1  # including embedding layer
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size

        if self.merge == 'linear':
            self.scale = ScalarMix(self.nb_layers)
            # self.weighing_params = nn.Parameter(torch.ones(self.num_layers), requires_grad=True)
        elif self.merge == 'attn':
            # all_enc_out range from 0-12 (including bert embedding)
            self.finetune_layer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            self.attn_proj = nn.ModuleList([nn.Linear(self.hidden_size, 1, bias=True) for _ in range(len(self.finetune_layer))])
            for m in self.attn_proj:
                # nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

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
            last_enc_out, _, all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)
        else:
            with torch.no_grad():
                last_enc_out, _, all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)

        if self.merge == 'linear':
            enc_out = self.scale(all_enc_outs[-self.nb_layers:])  # (bz, seq_len, 768)

            # encoded_repr = 0
            # soft_weight = F.softmax(self.weighing_params, dim=0)
            # for i in range(self.nb_layers):
            #     encoded_repr += soft_weight[i] * all_enc_outs[i]
            # enc_out = encoded_repr
        elif self.merge == 'mean':
            top_enc_outs = all_enc_outs[-self.nb_layers:]
            enc_out = sum(top_enc_outs) / len(top_enc_outs)
            # enc_out = torch.stack(tuple(top_enc_outs), dim=0).mean(0)
        elif self.merge == 'attn':
            # TODO: bert layer mix
            attn_score = torch.cat([fc(all_enc_outs[i]) for i, fc in zip(self.finetune_layer, self.attn_proj)], dim=-1)  # (B, N, 12)
            attn_score = F.softmax(attn_score, dim=-1)
            hidden_out = torch.stack([all_enc_outs[i] for i in self.finetune_layer], dim=2)
            # (B, N, 1, 12) @ (B, N, 12, 768) -> (B, N, 1, 768) -> (B, N, 768)
            enc_out = torch.matmul(attn_score.unsqueeze(-2), hidden_out).squeeze(-2)
        else:
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
        
    def avg_pooling(self, embedded_text_input: torch.FloatTensor,
                    offsets: torch.LongTensor):
        """
        Args:
            embedded_text_input : embedded subword embedding from pretrained models.
            offsets : subtoken indices.
        Returns:
            avg_pooling_result : avg_pooling result accroading to offsets.
        """
        batch_size, num_tokens, num_max_subwords = offsets.size()
        batch_index = torch.arange(batch_size, dtype=offsets.dtype, device=offsets.device).view(-1, 1, 1)
        token_index = torch.arange(num_tokens, dtype=offsets.dtype, device=offsets.device).view(1, -1, 1)
        _, num_total_subwords, hidden_size = embedded_text_input.size()
        embedded_text_input = embedded_text_input.unsqueeze(1).expand(batch_size, num_tokens, num_total_subwords, hidden_size)

        subwords_mask = offsets.ne(-1).sum(2)
        # 0 division
        padding_mask = subwords_mask.eq(0)
        # shape(batch_size, num_token)
        divisor = (subwords_mask + padding_mask).unsqueeze(2)

        offsets_mask = offsets.eq(-1).unsqueeze(3).expand(batch_size, num_tokens, num_max_subwords, hidden_size)
        # -1 padding will be out-of index
        offsets_index_with_zero = offsets.masked_fill(offsets.eq(-1), 0)
        token_repr = embedded_text_input[batch_index, token_index, offsets_index_with_zero]
        token_repr = token_repr.masked_fill_(offsets_mask, 0)
        token_repr = token_repr.sum(2)

        return token_repr / divisor
