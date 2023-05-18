import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from model.e2e_bert_domain_unseen_tagger import End2endSLUTagger, js_div
from transformers import BertModel
from utils.snips_e2ebert_domain_unseen_datautil import domain2slot
import copy
import collections


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index, reduction, label_smoothing):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.lb_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, logits, label):
        """
        Args:
            logits (_type_): size(B, N, num_type)
            label (_type_): size(B, N)
            mask (_type_): size(B, N)
        Returns:
            _type_: smoothing loss
        """
        with torch.no_grad():
            num_classes = logits.size(-1)
            label = label.clone().detach()
            ignore = label.eq(self.ignore_index)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smoothing, self.lb_smoothing / (num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                        lb_neg).scatter_(2, label.unsqueeze(-1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=-1)
        loss[ignore] = 0
        if self.reduction == 'batchmean':
            loss = loss.sum() / logits.size(0)
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def filter_param(inst):
    filter_ = ['hidden2boundary', 'hidden_proj', 'seq_encoder']
    if any(param in inst[0] for param in filter_):
        return True
    else:
        return False


class TTT(nn.Module):
    def __init__(self, vocabs, model_ckpt_path, args):
        super(TTT, self).__init__()
        self.model = End2endSLUTagger(
            vocabs,
            bert_embed_dim=768,
            num_bound=len(vocabs['binary']),
            num_bert_layer=4,
            dropout=args.dropout,
            use_cl=False,
            cl_type='cosine',
            cl_temperature=0.5,
            bert_model_path=args.bert_path
        ).to(args.device)
        self.args = args
        self.temperature = 1.0
        self.steps = 2
        self.vocabs = vocabs

        self.model.requires_grad_(False)
        finetune_params = list(filter(filter_param, [(nm, p) for nm, p in self.model.named_modules()]))
        for _, m in finetune_params:
            m.requires_grad_(True)
                
        optimizer_parameters = [
            {'params': [p for _, p in self.model.base_named_params() if p.requires_grad], 
             'weight_decay': 0.0, 'lr': 1e-4}
        ]

        # self.ce_loss = LabelSmoothingCrossEntropyLoss(ignore_index=0, reduction='none', label_smoothing=0.2)
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        self.reset_parameters(model_ckpt_path)
    
    def interpolate_weight(self, alpha=0.5):
        zs_bert = BertModel.from_pretrained('/home/tjuwlz/syj_project/zero-shot-slu/bert_model/bert-base-uncased', output_hidden_states=True).to(self.args.device).state_dict()
        ft_bert = self.model.bert.bert.state_dict()
        assert list(zs_bert.keys()) == list(ft_bert.keys())

        # TODO: interpolate on normalization layers
        inter_bert = {
            key: zs_bert[key] * (1 - alpha) + ft_bert[key] * alpha
            if 'LayerNorm' in key
            else ft_bert[key]
            for key in ft_bert.keys()
        }
        del zs_bert, ft_bert

        return inter_bert

    def reset_parameters(self, model_ckpt_path):
        """
        load model from best-dev model dict.
        """
        print(f'Loading the previous model states from {model_ckpt_path}')
        ckpt = torch.load(model_ckpt_path)
        # TODO: set othermodules except bert by requires grad = False?
        self.model.load_state_dict(deepcopy({k: v for k, v in ckpt['slu_model_state'].items() if 'tokencl' not in k}))
        
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.model.train()
        
        print('Previous best dev prf result is: %s' % ckpt['best_prf'])

    @torch.no_grad()
    def pred_pseudo_label(self, type_score, pred_bound, mask):  
        pred_type = type_score.argmax(dim=-1)
        seq_len = mask.sum(dim=-1)
        
        # boundary decode
        for i in range(seq_len.size(0)):
            prev = None
            for j in range(seq_len[i]):
                # 0 is [PAD] index
                if pred_type[i, j] == 0:
                    continue
                if pred_bound[i, j] == self.vocabs['binary'].inst2idx('B'):
                    prev = pred_type[i, j]
                elif pred_bound[i, j] == self.vocabs['binary'].inst2idx('I'):
                    if prev is None:
                        continue
                    else:
                        pred_type[i, j] = prev
        
        return pred_type.detach()
        
    @torch.enable_grad()
    def forward(self, bert_inp, unseen_inp, num_type, num_unseen_type, O_tag_idx, mask=None):
        # TODO: self training
        
        bsz = bert_inp[0].size(0)
        bert_inp = [torch.cat([x, x]) for x in bert_inp]
        mask = torch.cat([mask, mask], dim=0)
        for _ in range(self.steps):
            bert_repr = self.model.bert(*bert_inp)
            label_embedding = bert_repr[:, 1: num_type+1]
            token_repr = bert_repr[:, num_type+1:]
                
            # if num_unseen_type > 0:
            #     unseen_label_embedding = self.model.bert(*unseen_inp)[:, 1:]

            # TODO: R-drop
            enc_out, _ = self.model.seq_encoder(token_repr, non_pad_mask=mask.cpu())
            
            boundary_score = self.model.hidden2boundary(enc_out)            
            pred_bound = self.model.tag_decode(boundary_score, mask)
            bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), self.model.bound_embedding.weight)

            type_logits = self.model.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
            type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding.detach(), p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
            type_mask = (pred_bound.ne(O_tag_idx) * mask).float()
            
            # pseudo_labels = type_score.data.argmax(dim=-1)
            # loss = self.ce_loss(type_score, pseudo_labels)
            # B_mask = (mask * pred_bound.eq(self.vocabs['binary'].inst2idx('B'))).float()
            # loss = torch.sum(loss * B_mask, dim=-1).sum() / B_mask.sum()
            loss = 0.
            
            # # entropy loss
            # unseen_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(unseen_label_embedding.detach(), p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
            # loss = (-F.softmax(unseen_score, dim=-1) * F.log_softmax(unseen_score, dim=-1)).sum(-1)
            # loss = torch.sum(loss * type_mask, dim=-1).mean()

            # # TODO: filter padding and 'O' idx
            # filter_idx = type_mask.view(-1) > 0
            # bscore = type_score.view(-1, num_unseen_type)[filter_idx].contiguous()

            # ttt_loss = (-F.softmax(type_score, dim=-1) * F.log_softmax(type_score, dim=-1)).sum(-1)
            # # batch mean
            # ttt_loss = torch.sum(ttt_loss * type_mask, dim=-1).mean()
            
            # ttt_loss = F.cross_entropy(type_score.transpose(1, 2).contiguous(), gold_slu, ignore_index=0, reduction='none')  # (B, L)
            # ttt_loss = torch.sum(ttt_loss * type_mask, dim=-1)
            # ttt_loss = torch.mean(ttt_loss)

            input_ = type_score[:bsz]
            target_ = type_score[bsz:]
            
            # b_num = (pred_bound.ne(O_tag_idx) * pred_bound.ne(I_tag_idx) * mask).float()
            # input_B = b_num[:bsz].sum()
            # target_B = b_num[bsz:].sum()
            
            # ttt_loss = 0.5 * F.kl_div(input_.log_softmax(dim=-1), target_.softmax(dim=-1), reduction='none') * type_mask[bsz:].unsqueeze(-1) / target_B + \
            #            0.5 * F.kl_div(target_.log_softmax(dim=-1), input_.softmax(dim=-1), reduction='none') * type_mask[:bsz].unsqueeze(-1) / input_B
            ttt_loss = F.mse_loss(input_, target_, reduction='none') * type_mask[:bsz].unsqueeze(-1) * type_mask[bsz:].unsqueeze(-1)
            loss += torch.sum(ttt_loss.sum(-1), dim=-1).mean()
                       
            loss.backward()
            print(f"ttt loss: {loss.item()}")
            self.optimizer.step()
            self.model.zero_grad()

        return loss.item()

    # @torch.enable_grad()
    # def forward(self, bert_inp, unseen_inp, num_type, num_unseen_type, O_tag_idx, mask=None):
    #     TODO: contrastive test time learning
    #     for _ in range(self.steps):
    #         bert_repr = self.model.bert(*bert_inp)

    #         label_embedding = bert_repr[:, 1: num_type+1]
    #         token_repr = bert_repr[:, num_type+1:]

    #         enc_out, _ = self.model.seq_encoder(token_repr, non_pad_mask=mask.cpu())

    #         boundary_score = self.model.hidden2boundary(enc_out)

    #         pred_bound = self.model.tag_decode(boundary_score, mask)
    #         bound_embed = self.model.bound_embedding(pred_bound)

    #         type_logits = self.model.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
    #         type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding.detach(), p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
    #         type_mask = (pred_bound.ne(O_tag_idx) * mask).float()

    #         # filter padding and 'O' idx
    #         filter_idx = type_mask.view(-1) > 0
    #         # compute filter score
    #         filter_score = type_score.view(-1, num_type)[filter_idx].contiguous()
    #         filter_embedding = token_repr.contiguous().view(-1, self.args.emb_dim)[filter_idx]
    #         # shape (num_filter_samples,)
    #         filter_label = filter_score.data.argmax(dim=-1)

    #         # proj head
    #         # filter_embedding = self.model.tokencl.output_embedder_mu(filter_embedding)
    #         num_filter_samples, emb_dim = filter_embedding.size()
    #         # eg. (1, 2, 3, 1, 2, 3, ...)
    #         repeat_embedding = filter_embedding.unsqueeze(0).expand(num_filter_samples, num_filter_samples, emb_dim).contiguous().view(num_filter_samples * num_filter_samples, -1)
    #         # eg. (1, 1, 2, 2, 3, 3, ...)
    #         inter_embedding = filter_embedding.unsqueeze(1).expand(num_filter_samples, num_filter_samples, emb_dim).contiguous().view(num_filter_samples * num_filter_samples, -1)
    #         sim_score = F.cosine_similarity(inter_embedding, repeat_embedding, dim=-1).view(num_filter_samples, num_filter_samples)

    #         repeat_labels = filter_label.unsqueeze(0).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)
    #         inter_labels = filter_label.unsqueeze(1).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)

    #         denominator_mask = torch.all(repeat_embedding != inter_embedding, dim=-1).view(num_filter_samples, num_filter_samples).float()
    #         numerator_mask = denominator_mask * (repeat_labels == inter_labels).view(num_filter_samples, num_filter_samples).float()
           
    #         loss_all = torch.exp(sim_score / self.ttt_temperature)
    #         cnts = torch.sum(numerator_mask, dim = 1)
    #         loss_num = torch.sum(loss_all * numerator_mask, dim = 1)
    #         loss_denom = torch.sum(loss_all * denominator_mask, dim = 1)

    #         nonzero_indexes = torch.where(cnts > 0)
    #         loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]

    #         ttt_loss = torch.mean(-torch.log(loss_num) + torch.log(loss_denom) + torch.log(cnts))
            
    #         ttt_loss.backward()
    #         self.optimizer.step()
    #         self.model.zero_grad()

    #     return ttt_loss.item()

    @torch.no_grad()
    def evaluate(self, bert_inp, num_type, num_unseen_type, mask=None):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        '''
        bert_repr = self.model.bert(*bert_inp)
        label_embedding = bert_repr[:, 1: num_type+1]
        token_repr = bert_repr[:, num_type+1:]

        enc_out, _ = self.model.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        boundary_score = self.model.hidden2boundary(enc_out)

        pred_bound = self.model.tag_decode(boundary_score, mask)
        bound_embed = self.model.bound_embedding(pred_bound)

        type_logits = self.model.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
        type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
        
        pred_type = type_score.data.argmax(dim=-1)
        return pred_bound, pred_type


class TTA(nn.Module):
    def __init__(self, vocabs, model_ckpt_path, args, warmup_supports=None, warmup_labels=None, warmup_ent=None, filter_K=5):
        super(TTA, self).__init__()
        self.model = End2endSLUTagger(
            vocabs,
            bert_embed_dim=768,
            num_bound=len(vocabs['binary']),
            num_bert_layer=4,
            dropout=args.dropout,
            use_cl=False,
            bert_model_path=args.bert_path
        ).to(args.device)
        self.args = args
        self.temperature = 1.0
        self.steps = 3

        # calculate intial parameters
        self.supports = warmup_supports
        self.labels = warmup_labels
        self.ent = warmup_ent
        self.filter_K = filter_K

        optimizer_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()
                        if 'LayerNorm' in n and p.requires_grad], 
             'weight_decay': 0.0, 'lr': self.args.bert_lr},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)

        self.reset_parameters(model_ckpt_path)
    
    def reset_parameters(self, model_ckpt_path):
        """
        load model from best-dev model dict.
        """
        print(f'Loading the previous model states from {model_ckpt_path}')
        ckpt = torch.load(model_ckpt_path)
        # TODO: set othermodules except bert by requires grad = False?
        self.model.load_state_dict(deepcopy({k: v for k, v in ckpt['slu_model_state'].items() if 'tokencl' not in k}))
        
        self.model.zero_grad()
        self.model.train()
        print('Previous best dev prf result is: %s' % ckpt['best_prf'])

    def select_support(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.labels.size(-1)):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    @torch.enable_grad()
    def forward(self, i, bert_inp, num_type, O_tag_idx, mask=None):
        bert_repr = self.model.bert(*bert_inp)

        label_embedding = bert_repr[:, 1: num_type+1]
        token_repr = bert_repr[:, num_type+1:]

        # warmup support set.
        if i == 0:
            warmup_prob = torch.matmul(F.normalize(label_embedding, p=2, dim=-1), F.normalize(label_embedding.detach(), p=2, dim=-1).transpose(-1, -2)) / self.temperature
            self.supports = label_embedding.contiguous().view(-1, token_repr.size(-1)).clone()
            self.labels = F.one_hot(warmup_prob.argmax(-1).view(-1), num_classes=num_type).float()
            self.ent = (-F.softmax(warmup_prob, dim=-1) * F.log_softmax(warmup_prob, dim=-1)).sum(-1).view(-1)

        enc_out, _ = self.model.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        boundary_score = self.model.hidden2boundary(enc_out)

        pred_bound = self.model.tag_decode(boundary_score, mask)
        bound_embed = self.model.bound_embedding(pred_bound)

        type_logits = self.model.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())

        # predict label
        supports, labels = self.select_support()
        supports = F.normalize(supports, dim=-1)
        labels = F.normalize(labels, p=1, dim=0)
        weights = supports.transpose(0, 1).contiguous() @ labels
        type_score = torch.matmul(F.normalize(type_logits, dim=-1), weights.unsqueeze(0)) / self.temperature  # (B, N, D) x (1, D, K)
        type_mask = (pred_bound.ne(O_tag_idx) * mask).float()

        # calculate entropy
        ent_loss = (-F.softmax(type_score, dim=-1) * F.log_softmax(type_score, dim=-1)).sum(2)
        ent_loss = torch.mean((ent_loss * type_mask).sum(1))
        # # TODO: filter padding and 'O' idx
        # filter_idx = type_mask.view(-1) > 0
        # mscore = type_score.view(-1, num_type)[filter_idx].contiguous().mean(0)
        # g_ent = (-F.softmax(mscore, dim=-1) * F.log_softmax(mscore, dim=-1)).sum()
        # ent_loss -= g_ent

        # backward and update parameters
        ent_loss.backward()
        self.optimizer.step()
        self.model.zero_grad()

        return ent_loss.item()
    
    @torch.no_grad()
    def evaluate(self, i, bert_inp, num_type, O_tag_idx, mask=None):
        bert_repr = self.model.bert(*bert_inp)

        label_embedding = bert_repr[:, 1: num_type+1]
        token_repr = bert_repr[:, num_type+1:]

        enc_out, _ = self.model.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        boundary_score = self.model.hidden2boundary(enc_out)

        pred_bound = self.model.tag_decode(boundary_score, mask)
        bound_embed = self.model.bound_embedding(pred_bound)

        type_logits = self.model.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())

        # update support set during evaluation
        type_score = torch.matmul(F.normalize(type_logits, dim=-1), label_embedding.transpose(1, 2).contiguous()) / self.temperature  # (B, N, D) x (1, D, K)
        type_mask = (pred_bound.ne(O_tag_idx) * mask).float()

        pred_type = F.one_hot(type_score.argmax(dim=-1).view(-1), num_classes=num_type).float()
        # filter out tokens with padding or O label
        padding_o_filter_idx = type_mask.view(-1) > 0

        self.supports = torch.cat([self.supports, type_logits.view(-1, token_repr.size(-1))[padding_o_filter_idx]])
        self.labels = torch.cat([self.labels, pred_type[padding_o_filter_idx]])
        ent = (-F.softmax(type_score, dim=-1) * F.log_softmax(type_score, dim=-1)).sum(-1)
        self.ent = torch.cat([self.ent, ent.view(-1)[padding_o_filter_idx]])

        # predict label
        supports, labels = self.select_support()
        supports = F.normalize(supports, dim=-1)
        labels = F.normalize(labels, p=1, dim=0)
        weights = supports.transpose(0, 1).contiguous() @ labels
        return pred_bound, torch.matmul(F.normalize(type_logits, dim=-1), weights.unsqueeze(0)).argmax(dim=-1)


class TTDistill(nn.Module):
    def __init__(self, vocabs, model_ckpt_path, args):
        super(TTDistill, self).__init__()
        self.teacher = End2endSLUTagger(
            vocabs,
            bert_embed_dim=768,
            num_bound=len(vocabs['binary']),
            num_bert_layer=4,
            dropout=args.dropout,
            use_cl=False,
            cl_type='cosine',
            cl_temperature=0.5,
            bert_model_path=args.bert_path
        ).to(args.device)
        
        self.student = copy.deepcopy(self.teacher)
        
        self.args = args
        self.temperature = 1.0
        self.steps = 1
        self.vocabs = vocabs
        
        self.teacher.requires_grad_(False)
        self.student.requires_grad_(False)
        stu_finetune_params = list(filter(filter_param, [(nm, p) for nm, p in self.student.named_modules()]))
        for _, m in stu_finetune_params:
            m.requires_grad_(True)
            
        # tea_finetune_params = list(filter(filter_param, [(nm, p) for nm, p in self.teacher.named_modules()]))
        # for _, m in tea_finetune_params:
        #     m.requires_grad_(True)
                
        optimizer_parameters = [
            {'params': [p for _, p in self.student.base_named_params() if p.requires_grad], 
             'weight_decay': 0.0, 'lr': 1e-4}
        ]

        self.ce_loss = LabelSmoothingCrossEntropyLoss(ignore_index=0, reduction='none', label_smoothing=0.2)
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        self.reset_parameters(model_ckpt_path)

    def reset_parameters(self, model_ckpt_path):
        """
        load model from best-dev model dict.
        """
        print(f'Loading the previous model states from {model_ckpt_path}')
        ckpt = torch.load(model_ckpt_path)
        # TODO: set othermodules except bert by requires grad = False?
        self.model_cache = deepcopy({k: v for k, v in ckpt['slu_model_state'].items() if 'tokencl' not in k})
        self.teacher.load_state_dict(self.model_cache)
        self.teacher.seq_encoder.lstm.flatten_parameters()
        self.student.load_state_dict(self.model_cache)
        self.student.seq_encoder.lstm.flatten_parameters()  
        
        self.teacher.zero_grad()
        self.student.zero_grad()
        self.optimizer.zero_grad()
        self.teacher.train()
        self.student.train()
        
        print('Previous best dev prf result is: %s' % ckpt['best_prf'])

    def onepass_reset(self):
        self.student.load_state_dict(self.model_cache)
        self.student.seq_encoder.lstm.flatten_parameters()
        self.optimizer.zero_grad()
        self.optimizer.state = collections.defaultdict(dict)
    
    @torch.enable_grad()
    def forward(self, bert_inp, unseen_inp, num_type, num_unseen_type, O_tag_idx, mask=None):
        
        def get_type_logits(model, bert_inp, num_type, mask, temperature):
            bert_repr = model.bert(*bert_inp)
            label_embedding = bert_repr[:, 1: num_type+1]
            token_repr = bert_repr[:, num_type+1:]
            enc_out, _ = model.seq_encoder(token_repr, non_pad_mask=mask.cpu())

            boundary_score = model.hidden2boundary(enc_out)
            pred_bound = model.tag_decode(boundary_score, mask)
            bound_embed = torch.matmul(F.softmax(boundary_score, dim=-1), model.bound_embedding.weight)
            
            type_logits = model.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
            type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding.detach(), p=2, dim=-1).transpose(-1, -2)) / temperature  # (B, N, D) x (B, D, K)
            
            return type_score, pred_bound

        def update_param_with_ema(student, teacher, alpha_teacher=0.999):
            for _student, _teacher in zip(student.parameters(), teacher.parameters()):
                if _teacher.requires_grad:
                    _teacher.data[:] = alpha_teacher * _teacher[:].data[:] + (1 - alpha_teacher) * _student[:].data[:]
            return teacher
        
        # TODO: without updating teacher model
        with torch.no_grad():
            teacher_score, _ = get_type_logits(self.teacher, bert_inp, num_type, mask, temperature=self.temperature)
                
        for _ in range(self.steps):
            # TODO: EMA update teacher model
            # with torch.no_grad():
            #     teacher_score, _ = get_type_logits(self.teacher, bert_inp, num_type, mask, temperature=self.temperature)
            student_score, pred_bound = get_type_logits(self.student, bert_inp, num_type, mask, temperature=self.temperature)
            type_mask = (pred_bound.ne(O_tag_idx) * mask).float()

            # ttt_loss = F.kl_div(student_score.log_softmax(dim=-1), teacher_score.softmax(dim=-1).detach(), reduction='none')
            ttt_loss = F.mse_loss(student_score.softmax(dim=-1), teacher_score.softmax(dim=-1).detach(), reduction='none')
            ttt_loss = torch.sum(ttt_loss * type_mask.unsqueeze(-1), dim=-1).sum(-1).mean()
            
            pseudo_labels = student_score.data.argmax(dim=-1)
            loss = self.ce_loss(student_score, pseudo_labels)
            B_mask = (mask * pred_bound.eq(self.vocabs['binary'].inst2idx('B'))).float()
            loss = torch.sum(loss * B_mask, dim=-1).sum() / B_mask.sum()
            ttt_loss += loss

            self.student.zero_grad()
            ttt_loss.backward()
            self.optimizer.step()
            
            # update teacher with moving average.
            # self.teacher = update_param_with_ema(self.student, self.teacher, alpha_teacher=0.5)

        return ttt_loss.item()

    @torch.no_grad()
    def evaluate(self, bert_inp, num_type, num_unseen_type, mask=None):
        '''Decoding Rules:
        以边界标签优先，如果预测为O，则不管类别标签是什么，最终标签是O；
        如果预测的是其他类型，则将边界标签和类别标签进行组合，进而评估最终性能
        '''
        bert_repr = self.student.bert(*bert_inp)

        label_embedding = bert_repr[:, 1: num_type+1]
        token_repr = bert_repr[:, num_type+1:]

        enc_out, _ = self.student.seq_encoder(token_repr, non_pad_mask=mask.cpu())
        boundary_score = self.student.hidden2boundary(enc_out)

        pred_bound = self.student.tag_decode(boundary_score, mask)
        bound_embed = self.student.bound_embedding(pred_bound)

        type_logits = self.student.hidden_proj(torch.cat((token_repr, bound_embed), dim=-1).contiguous())
        type_score = torch.matmul(F.normalize(type_logits, p=2, dim=-1), F.normalize(label_embedding, p=2, dim=-1).transpose(-1, -2)) / self.temperature  # (B, N, D) x (B, D, K)
        
        pred_type = type_score.data.argmax(dim=-1)
        return pred_bound, pred_type