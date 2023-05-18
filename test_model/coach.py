import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.test_tagger import SLUSeqTagger, BinaryModel, TemplateEncoder
from modules.embedding import PretrainedEmbedding
from config.slu_conf import args_config
from utils.dataset import DataLoader
from utils.coach_datautil import load_data, create_vocab, batch_variable, save_to
from data.snips.generate_slu_emb import domain2slot
from utils.conlleval import evaluate
import copy
from logger.logger import logger
import pickle as pkl


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        print(self.args)
        
        self.all_dataset = load_data()
        self.tgt_dm = args.tgt_dm
        self.n_sample = args.n_sample
        
        self.train_set = []
        self.val_set = []
        self.test_set = []
        for dm, insts in self.all_dataset.items():
            if dm != self.tgt_dm:
                self.train_set.extend(insts)
            else:
                self.train_set.extend(insts[: self.n_sample])
                self.val_set.extend(insts[self.n_sample: 500])
                self.test_set.extend(insts[500:])
 
        print('train data size:', len(self.train_set))
        print('validate data size:', len(self.val_set))
        print('test data size:', len(self.test_set))


        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.val_set, batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size, shuffle=False)

        # if not os.path.exists("data/snips/cache/vocab_ckpt.pkl"):
        # self.vocabs = create_vocab(self.all_dataset)
            # save_to(args.vocab_chkp, self.vocabs)
        # else:
        with open("data/snips/cache/vocab_ckpt.pkl", 'rb') as f:
            self.vocabs = pkl.load(f)
            print(self.vocabs["binary"]._inst2idx)
            print(self.vocabs["slot"]._inst2idx)
            print(self.vocabs["domain"]._inst2idx)

        self.word_embedding = PretrainedEmbedding(vocab=self.vocabs["token"],
                                                  emb_file=self.args.emb_file, 
                                                  num_words=len(self.vocabs["token"]),
                                                  emb_dim=self.args.emb_dim
                                                ).to(args.device)

        self.binary_model = BinaryModel(copy.deepcopy(self.word_embedding), 
                                self.args.emb_dim, 
                                self.args.hidden_size, 
                                self.args.num_rnn_layer,
                                len(self.vocabs["binary"]._inst2idx), 
                                self.args.bidirectional,
                                self.args.freeze_emb,
                                self.args.dropout
                            ).to(args.device)
                            
        self.slu_model = SLUSeqTagger(
                            self.vocabs,
                            args.slot_emb_file, 
                            self.args.hidden_size, 
                            self.args.num_rnn_layer, 
                            self.args.bidirectional
                        ).to(args.device)
        
        self.template_enc = TemplateEncoder(
                                copy.deepcopy(self.word_embedding), 
                                self.args.emb_dim, 
                                self.args.hidden_size, 
                                self.args.num_rnn_layer, 
                                self.args.bidirectional, 
                                self.args.dropout,
                                self.args.freeze_emb
                            ).to(args.device)

        print(self.word_embedding)
        print(self.binary_model)
        print(self.slu_model)
        print(self.template_enc)
        total_params = sum([p.numel() for gen in [self.binary_model.parameters(), self.slu_model.parameters(),
                                                self.template_enc.parameters()] for p in gen if p.requires_grad])
        print("Training %dM trainable parameters..." % (total_params/1e6))

        self.slu_tag_loss = nn.CrossEntropyLoss(reduction='mean')
        self.tr_loss = nn.MSELoss(reduction='mean')
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_parameters = [
        #     {'params': [p for _, p in self.binary_model.named_parameters() if p.requires_grad],
        #      'weight_decay': self.args.weight_decay, 'lr': self.args.lr},
        #     {'params': [p for _, p in self.slu_model.named_parameters() if p.requires_grad],
        #      'weight_decay': self.args.weight_decay, 'lr': self.args.lr},
        #     {'params': [p for _, p in self.template_enc.named_parameters() if p.requires_grad],
        #      'weight_decay': self.args.weight_decay, 'lr': self.args.lr}
        # ]
        model_parameters = [
            {'params': self.binary_model.parameters(), 'lr': 1e-3},
            {'params': self.slu_model.parameters()},
            {'params': self.template_enc.parameters()}
        ]
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.args.lr)
        # self.optimizer = torch.optim.AdamW(optimizer_bert_parameters, lr=self.args.bert_lr, eps=self.args.eps)
        # max_step = len(self.train_loader) * self.args.epoch
        # self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 10, t_total=max_step)

    def train_epoch(self, ep=0):
        self.binary_model.train()
        self.slu_model.train()
        self.template_enc.train()
        
        t1 = time.time()
        loss_tr = {"bio_loss": 0., "sent_slu_loss": 0., "tem0_loss": 0., "tem1_loss": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            tag_repr, lstm_reprs = self.binary_model(batch.tokens, batch.token_mask)
            bio_loss = self.binary_model.compute_loss(tag_repr, batch.bio_label, mask=batch.token_mask)

            self.optimizer.zero_grad()
            bio_loss.backward(retain_graph=True)
            loss_tr["bio_loss"] = bio_loss.item()
            self.optimizer.step()
            
            loss_tr["sent_slu_loss"] = 0.
            batch_slu_loss = 0.
            _, lstm_reprs = self.binary_model(batch.tokens, batch.token_mask)
            pred_slotname_list, gold_slotname_list = self.slu_model(batch.domains, lstm_reprs, bio_gold=batch.bio_label, slu_gold=batch.slu_label)
            for j, (pred, gold) in enumerate(zip(pred_slotname_list, gold_slotname_list)):
                assert pred.size(0) == gold.size(0)
                slu_loss = self.slu_tag_loss(pred, gold)
                batch_slu_loss += slu_loss / len(pred_slotname_list)
            self.optimizer.zero_grad()
            batch_slu_loss.backward(retain_graph=True)
            self.optimizer.step()
            loss_tr["sent_slu_loss"] = batch_slu_loss.item()

            # _, lstm_reprs = self.binary_model(batch.tokens, batch.token_mask)
            # input_repr, templates_repr = self.template_enc(batch.templates, batch.token_mask, batch.template_mask, lstm_reprs)
            # input_repr = input_repr.detach()
            # tem0_loss = self.tr_loss(input_repr, templates_repr[:, 0])
            # loss_tr["tem0_loss"] = tem0_loss.item()
            # tem1_loss = -self.tr_loss(input_repr, templates_repr[:, 1])
            # loss_tr["tem1_loss"] = tem1_loss.item()
            # tem2_loss = -self.tr_loss(input_repr, templates_repr[:, 2])
            
            # input_repr.requires_grad = True
            # self.optimizer.zero_grad()
            # tem0_loss.backward(retain_graph=True)
            # tem1_loss.backward(retain_graph=True)
            # tem2_loss.backward(retain_graph=True)
            # self.optimizer.step()
            
            # if ep > 3:
            #     templates_repr = templates_repr.detach()
            #     input0_loss = self.tr_loss(input_repr, templates_repr[:, 0])
            #     input1_loss = -self.tr_loss(input_repr, templates_repr[:, 1])
            #     input2_loss = -self.tr_loss(input_repr, templates_repr[:, 2])
            #     templates_repr.requires_grad = True
            #     self.optimizer.zero_grad()
            #     input0_loss.backward(retain_graph=True)
            #     input1_loss.backward(retain_graph=True)
            #     input2_loss.backward(retain_graph=True)
            #     self.optimizer.step()
            tem0_loss = torch.tensor([0.], device=batch.tokens.device)
            tem1_loss = torch.tensor([0.], device=batch.tokens.device)

            logger.info('[Epoch %d] Iter%d time cost: %.4fs, bio loss: %.4f, slu_loss: %.4f, tem0_loss: %.4f, tem1_loss: %.4f' % \
                (ep, i, time.time() - t1, bio_loss.item(), slu_loss.item(), tem0_loss.item(), tem1_loss.item()))

        return loss_tr


    def save_states(self, save_path, best_test_metric=None):
        #bert_optim_state = {'optimizer': self.bert_optimizer.state_dict(),
        #                    'scheduler': self.bert_scheduler.state_dict()}
        #if os.path.exists(save_path):
        #    os.remove(save_path)

        self.word_embedding.zero_grad()
        self.binary_model.zero_grad()
        self.slu_model.zero_grad()
        self.template_enc.zero_grad()
        check_point = {'word_embedding': self.word_embedding.state_dict(),
                       'binary_model_state': self.binary_model.state_dict(),
                       'slu_model_state': self.slu_model.state_dict(),
                       'template_enc': self.template_enc.state_dict(),
                       'args': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')

    # def restore_states(self, load_path):
    #     ckpt = torch.load(load_path)
    #     # torch.set_rng_state(ckpt['rng_state'])
    #     self.model.load_state_dict(ckpt['model_state'])
    #     self.optimizer.load_state_dict(ckpt['optimizer_state'])
    #     self.meta_opt.load_state_dict(ckpt['meta_opt_state'])
    #     self.args = ckpt['args_settings']
    #     self.model.zero_grad()
    #     self.optimizer.zero_grad()
    #     self.meta_opt.zero_grad()
    #     self.model.train()
    #     logger.info('Loading the previous model states ...')
    #     print('Previous best prf result is: %s' % ckpt['best_prf'])

    def train(self):
        patient = 0
        best_dev_metric, best_test_metric = dict(), dict()
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.train_epoch(ep)

            dev_metric = self.evaluate(self.dev_loader)
            if dev_metric['slu_f'] > best_dev_metric.get('slu_f', 0):
                best_dev_metric = dev_metric
                # self.save_states(self.args.model_ckpt, best_dev_metric)
                test_metric = self.evaluate(self.test_loader)
                if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
                   best_test_metric = test_metric
                #    self.save_states(self.args.model_ckpt, best_test_metric)
                patient = 0
            else:
                patient += 1

            if patient >= self.args.patient:
                print(f"======early stopping with patient:{patient}======")
                break
            logger.info('[Epoch %d] train loss: %s, patient: %d, dev_metric: %s, test_metric: %s' %\
                (ep, train_loss, patient, best_dev_metric, test_metric))

        test_metric = self.evaluate(self.test_loader)
        if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
           best_test_metric = test_metric
        #    self.save_states(self.args.model_chkp, best_test_metric)
        logger.info(f'Final Test Metric: {best_dev_metric}, Test Metric: {best_test_metric}')
        return test_metric


    def evaluate(self, test_loader):
        self.binary_model.eval()
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        with torch.no_grad():
            for _, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                tag_repr, lstm_reprs = self.binary_model(batch.tokens, mask=batch.token_mask)
                batch_bio_pred = self.binary_model.crf.decode(tag_repr, mask=batch.token_mask)
                # batch_bio_pred_ids = torch.zeros_like(batch.bio_label).fill_(self.vocabs["binary"].inst2idx('<pad>'))
                
                # 将pred label转化为pred id
                batch_bio_pred_ids = pad_sequence(batch_bio_pred, batch_first=True, padding_value=self.vocabs["binary"].inst2idx('<pad>'))
                # for j in range(len(batch_bio_pred)):
                #     batch_bio_pred_ids[j, :len(batch_bio_pred[j])] = \
                #         torch.tensor([x for x in batch_bio_pred[j]], dtype=torch.long, device=batch.tokens.device)
                batch_slu_pred = self.slu_model.evaluate(batch.domains, lstm_reprs, 
                                                bio_pred=batch_bio_pred_ids)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, (bio_pred, slu_pred) in enumerate(zip(batch_bio_pred, batch_slu_pred)):
                    bio_gold_tags.extend(batcher[j].binary_tags)
                    slu_gold_tags.extend(batcher[j].slu_tags)
                    if slu_pred == None:
                        assert sum(bio_pred == self.vocabs["binary"].inst2idx('B')) == 0
                    else:
                        assert len(slu_pred) == sum(bio_pred == self.vocabs["binary"].inst2idx('B'))
                    slot_based_on_domain = domain2slot[self.vocabs["domain"].idx2inst(batch.domains[j].item())]
                    if slu_pred is not None:
                        slu_pred = slu_pred.argmax(dim=-1)
                    k = -1
                    for tag in bio_pred.cpu().tolist():
                        bio_pred_tags.append(self.vocabs["binary"].idx2inst(tag))
                        if tag == self.vocabs["binary"].inst2idx('O'):
                            slu_pred_tags.append('O')
                        elif tag == self.vocabs["binary"].inst2idx('B'):
                            k += 1
                            slu_name = slot_based_on_domain[slu_pred[k].item()]
                            if slu_name in slot_based_on_domain:
                                slu_pred_tags.append('B-' + slu_name)
                            else:
                                slu_pred_tags.append('O')
                        elif tag == self.vocabs["binary"].inst2idx('I'):
                            if k == -1:
                                slu_pred_tags.append('O')
                            else:
                                slu_name = slot_based_on_domain[slu_pred[k].item()]
                                if slu_name in slot_based_on_domain:
                                    slu_pred_tags.append('I-' + slu_name)
                                else:
                                    slu_pred_tags.append('O')
                        else:
                            slu_pred_tags.append('O')
        bio_p, bio_r, bio_f = evaluate([x + '-1' if x != 'O' else x for x in bio_gold_tags], 
                                       [x + '-1' if x != 'O' else x for x in bio_pred_tags], verbose=False)
        slu_p, slu_r, slu_f = evaluate(slu_gold_tags, slu_pred_tags, verbose=False) 
        return {"bio_p": bio_p, "bio_r": bio_r, "bio_f": bio_f, \
                "slu_p": slu_p, "slu_r": slu_r, "slu_f": slu_f}


def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    # random_seeds = [1314, 1357, 2789, 3391, 3407, 4553, 5919]
    
    final_res = {'p': [], 'r': [], 'f': []}
    seed = 1314
    set_seeds(seed)
    print(f"set seed: {seed}")
    trainer = Trainer(args, data_config=None)
    prf = trainer.train()
    #final_res['p'].append(prf['p'])
    #final_res['r'].append(prf['r'])
    #final_res['f'].append(prf['f'])

    '''
    logger.info('Final Result: %s' % final_res)
    final_p = sum(final_res['p']) / len(final_res['p'])
    final_r = sum(final_res['r']) / len(final_res['r'])
    final_f = sum(final_res['f']) / len(final_res['f'])
    logger.info('Final P: %.4f, R: %.4f, F: %.4f' % (final_p, final_r, final_f))
    '''

# baseline model
# nohup python coach.py --tgt_dm AddToPlaylist --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_atp50.ckpt &> training_log/baseline_atp50.log &
# nohup python coach.py --tgt_dm BookRestaurant --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_br50.ckpt -lr 5e-4 &> training_log/baseline_br50.log &
# nohup python coach.py --tgt_dm GetWeather --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_gw50.ckpt &> training_log/baseline_gw50.log &
# nohup python coach.py --tgt_dm PlayMusic --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_pm50.ckpt &> training_log/baseline_pm50.log &
# nohup python coach.py --tgt_dm RateBook --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_rb50.ckpt &> training_log/baseline_rb50.log &
# nohup python coach.py --tgt_dm SearchCreativeWork --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_scw50.ckpt &> training_log/baseline_scw50.log &
# nohup python coach.py --tgt_dm SearchScreeningEvent --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_sse50.ckpt &> training_log/baseline_sse50.log &

# baseline zero-shot
# nohup python coach.py --tgt_dm AddToPlaylist --n_sample 0 --emb_file data/snips/cache/slu+slot_word_char.pkl --freeze_emb --model_ckpt ckpt/coach/baseline_atp0.ckpt &> training_log/coach/baseline_atp0.log &
# nohup python coach.py --cuda 6 --tgt_dm BookRestaurant -lr 5e-4 --epoch 300 --n_sample 0 --num_rnn_layer 2 --dropout 0.3 --emb_dim 400 --freeze_emb --tr --emb_file data/snips/cache/slu+slot_word_char.npy --slot_emb_file data/snips/cache/slot_word_char.dict --model_ckpt ckpt/debug.ckpt &> testlog/coach_test1_br0.log &
# nohup python coach.py --tgt_dm GetWeather --n_sample 0 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_gw0.ckpt &> training_log/baseline_gw0.log &
# nohup python coach.py --tgt_dm PlayMusic --n_sample 0 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_pm0.ckpt &> training_log/baseline_pm0.log &
# nohup python coach.py --tgt_dm RateBook --n_sample 0 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_rb0.ckpt &> training_log/baseline_rb0.log &
# nohup python coach.py --tgt_dm SearchCreativeWork --n_sample 0 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_scw0.ckpt &> training_log/baseline_scw0.log &
# nohup python coach.py --tgt_dm SearchScreeningEvent --n_sample 0 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/baseline_sse0.ckpt &> training_log/baseline_sse0.log &