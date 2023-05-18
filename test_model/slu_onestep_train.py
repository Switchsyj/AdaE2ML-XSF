import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.test_tagger import SLUSeqTagger, BinaryModel, TemplateEncoder, BatchSLUTagger, BatchSLULSTMTagger
from modules.embedding import PretrainedEmbedding
from config.slu_conf import args_config
from utils.dataset import DataLoader
from utils.coach_datautil import load_data, create_vocab, batch_variable, save_to
from data.snips.generate_slu_emb import domain2slot
from utils.conlleval import evaluate
import copy
import pickle as pkl
# from utils.tag_util import *
# from utils.conlleval import evaluate
# from utils.eval import calc_prf
# import torch.nn.utils as nn_utils
from logger.logger import logger

# TODO: set random seed.

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

        if not os.path.exists("data/snips/cache/vocab_ckpt.pkl"):
            self.vocabs = create_vocab(self.all_dataset)
            save_to(args.vocab_chkp, self.vocabs)
        else:
            with open("data/snips/cache/vocab_ckpt.pkl", 'rb') as f:
                self.vocabs = pkl.load(f)
                print(self.vocabs["binary"]._inst2idx)
                print(self.vocabs["slot"]._inst2idx)
                print(self.vocabs["domain"]._inst2idx)

        self.word_embedding = PretrainedEmbedding("data/snips/cache/slu_word_char_embs_with_slotembs.npy",
                                                  len(self.vocabs["token"]._inst2idx), 
                                                  self.args.emb_dim,
                                                  self.vocabs["token"].inst2idx('<pad>')
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
        self.slu_model = BatchSLUTagger(
                            self.vocabs, 
                            args.bio_emb_dim,
                            args.slot_emb_file,
                            self.args.hidden_size,
                            self.args.num_rnn_layer,
                            self.args.bidirectional
                        ).to(args.device)
        # self.slu_model = BatchSLULSTMTagger(
        #                     self.vocabs, 
        #                     args.bio_emb_dim,
        #                     args.slot_emb_file,
        #                     self.args.hidden_size,
        #                     self.args.num_rnn_layer,
        #                     self.args.bidirectional
        #                 ).to(args.device)
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

        self.tr_loss = nn.MSELoss(reduction='mean')
        # no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for _, p in self.binary_model.named_parameters() if p.requires_grad],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lstmcrf_lr},
            {'params': [p for _, p in self.slu_model.named_parameters() if p.requires_grad],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lr},
            {'params': [p for _, p in self.template_enc.named_parameters() if p.requires_grad],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lr}
        ]
        self.optimizer = torch.optim.Adam(optimizer_parameters, lr=self.args.lr)
        # self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr)
        # max_step = len(self.train_loader) * self.args.epoch
        # self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 10, t_total=max_step)
        
    def train_epoch(self, ep=0):
        self.binary_model.train()
        self.slu_model.train()
        self.template_enc.train()
        
        t1 = time.time()
        loss_tr = {"bio_loss": 0., "slu_loss": 0., "tem0_loss": 0., "tem1_loss": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            tag_repr, lstm_reprs = self.binary_model(batch.tokens, batch.token_mask)
            # sent level
            bio_loss = self.binary_model.compute_loss(tag_repr, batch.bio_label, mask=batch.token_mask)
            loss = bio_loss
            loss_tr["bio_loss"] = bio_loss.item()
            

            bio_pred = self.binary_model.crf.decode(tag_repr, mask=batch.token_mask)
            slu_loss = self.slu_model(batch.domains, lstm_reprs, bio_pred=bio_pred, slu_gold=batch.slu_label, token_mask=batch.token_mask)
            # sent level
            loss += slu_loss / batch.tokens.size(0)
            loss_tr["slu_loss"] = slu_loss.item() / batch.tokens.size(0)
            
            input_repr, templates_repr = self.template_enc(batch.templates, batch.token_mask, batch.template_mask, lstm_reprs)
            input_repr = input_repr.detach()
            # sent level
            loss_tr['tem0_loss']
            tem0_loss = self.tr_loss(input_repr, templates_repr[:, 0])
            loss_tr['tem0_loss'] = tem0_loss.item()
            loss += tem0_loss
            tem1_loss = -self.tr_loss(input_repr, templates_repr[:, 1])
            loss_tr['tem1_loss'] = tem1_loss.item()
            loss -= tem1_loss
            loss -= self.tr_loss(input_repr, templates_repr[:, 2])
        
            input_repr.requires_grad = True
            
            
            if ep > 4:
                templates_repr = templates_repr.detach()
                loss += self.tr_loss(input_repr, templates_repr[:, 0])
                loss -= self.tr_loss(input_repr, templates_repr[:, 1])
                loss -= self.tr_loss(input_repr, templates_repr[:, 2])
                templates_repr.requires_grad = True
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            logger.info('[Epoch %d] Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f, slu_loss: %.4f, tem0_loss: %.4f, tem1_loss: %.4f' % \
                (ep, i, (time.time() - t1), loss.item(), loss_tr["bio_loss"], loss_tr["slu_loss"], loss_tr["tem0_loss"], loss_tr["tem1_loss"]))

        return loss_tr


    def save_states(self, save_path, best_test_metric=None):
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
        

    def train(self):
        patient = 0
        best_dev_metric, test_metric = dict(), dict()
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.train_epoch(ep)

            dev_metric = self.evaluate(self.dev_loader)
            if dev_metric['slu_f'] > best_dev_metric.get('slu_f', 0):
                best_dev_metric = dev_metric
                self.save_states(self.args.model_ckpt, best_dev_metric)
                test_metric = self.evaluate(self.test_loader)
                # if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
                #    best_test_metric = test_metric
                #    self.save_states(self.args.model_ckpt, best_test_metric)
                patient = 0
            else:
                patient += 1

            if patient >= self.args.patient:
                print(f"======early stopping with patient:{patient}======")
                break
            logger.info('[Epoch %d] train loss: %s, patient: %d, dev_metric: %s, test_metric: %s' %\
                (ep, train_loss, patient, best_dev_metric, test_metric))

        # test_metric = self.evaluate(self.test_loader)
        # if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
        #    best_test_metric = test_metric
        #    self.save_states(self.args.model_chkp, best_test_metric)
        logger.info(f'Final Test Metric: {best_dev_metric}, Test Metric: {test_metric}')
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
                batch_bio_pred_ids = torch.zeros_like(batch.bio_label).fill_(self.vocabs["binary"].inst2idx('<pad>'))
                
                # 将pred label转化为pred id
                for j in range(len(batch_bio_pred)):
                    batch_bio_pred_ids[j, :len(batch_bio_pred[j])] = \
                        torch.tensor([x for x in batch_bio_pred[j]], dtype=torch.long, device=batch.tokens.device)
                batch_slu_pred = self.slu_model.evaluate(batch.domains, lstm_reprs, 
                                                        bio_pred=batch_bio_pred_ids, token_mask=batch.token_mask)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, (bio_pred, slu_pred) in enumerate(zip(batch_bio_pred, batch_slu_pred)):
                    bio_gold_tags.extend([self.vocabs["binary"].idx2inst(x.item()) for x in batch.bio_label[j][:seq_len[j]]])
                    slu_gold_tags.extend([self.vocabs["slot"].idx2inst(x.item()) for x in batch.slu_label[j][:seq_len[j]]])

                    slu_pred = slu_pred[:len(bio_pred)].argmax(dim=-1)
                    slot_based_on_domain = domain2slot[self.vocabs["domain"].idx2inst(batch.domains[j].item())]
                    for k, tag in enumerate(bio_pred.cpu().tolist()):
                        bio_pred_tags.append(self.vocabs["binary"].idx2inst(tag))
                        if tag == self.vocabs["binary"].inst2idx('O'):
                            slu_pred_tags.append('O')
                        elif tag == self.vocabs["binary"].inst2idx('B'):
                            slu_pred_tags.append('B-' + slot_based_on_domain[slu_pred[k].item()])
                        elif tag == self.vocabs["binary"].inst2idx('I'):
                            slu_pred_tags.append('I-' + slot_based_on_domain[slu_pred[k].item()])
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

    # data_path = data_config('./config/data_path.json')

    random_seeds = [1314, 1357, 2789, 3391, 4553, 5919]
    final_res = {'p': [], 'r': [], 'f': []}
    for seed in random_seeds:
        set_seeds(seed)
        print(f"set seed: {seed}")
        trainer = Trainer(args, data_config=None)
        prf = trainer.train()
        #final_res['p'].append(prf['p'])
        #final_res['r'].append(prf['r'])
        #final_res['f'].append(prf['f'])
        break

    '''
    logger.info('Final Result: %s' % final_res)
    final_p = sum(final_res['p']) / len(final_res['p'])
    final_r = sum(final_res['r']) / len(final_res['r'])
    final_f = sum(final_res['f']) / len(final_res['f'])
    logger.info('Final P: %.4f, R: %.4f, F: %.4f' % (final_p, final_r, final_f))
    '''

# nohup python slu_onestep_train.py --cuda 0 --tgt_dm AddToPlaylist --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_atp50.ckpt -lr 1e-3 &> training_log/onestep_atp50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm BookRestaurant --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_br50.ckpt -lr 1e-3 &> training_log/onestep_br50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm GetWeather --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_gw50.ckpt -lr 1e-3 &> training_log/onestep_gw50.log &

### 2022.6.15
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm PlayMusic --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_pm50.ckpt -lr 1e-3 &> training_log/onestep_pm50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm RateBook --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_rb50.ckpt -lr 1e-3 &> training_log/onestep_rb50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm SearchCreativeWork --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_scw50.ckpt -lr 1e-3 &> training_log/onestep_scw50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm SearchScreeningEvent --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_sse50.ckpt -lr 1e-3 &> training_log/onestep_sse50.log &

### BatchSLULSTMTagger(2022.6.15)
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm AddToPlaylist --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_lstm_atp50.ckpt -lr 1e-3 &> training_log/onestep_lstm_atp50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm BookRestaurant --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_lstm_br50.ckpt -lr 1e-3 &> training_log/onestep_lstm_br50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm GetWeather --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_lstm_gw50.ckpt -lr 1e-3 &> training_log/onestep_lstm_gw50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm PlayMusic --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_lstm_pm50.ckpt -lr 1e-3 &> training_log/onestep_lstm_pm50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm RateBook --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_lstm_rb50.ckpt -lr 1e-3 &> training_log/onestep_lstm_rb50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm SearchCreativeWork --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_lstm_scw50.ckpt -lr 1e-3 &> training_log/onestep_lstm_scw50.log &
# nohup python slu_onestep_train.py --cuda 0 --tgt_dm SearchScreeningEvent --n_sample 50 --emb_file data/snips/cache/slu_word_char_embs_with_slotembs.npy --freeze_emb --model_ckpt ckpt/onestep_lstm_sse50.ckpt -lr 1e-3 &> training_log/onestep_lstm_sse50.log &
