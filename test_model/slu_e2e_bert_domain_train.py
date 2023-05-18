from collections import defaultdict
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.e2e_bert_domain_tagger import End2endSLUTagger
from config.slu_conf import args_config
from utils.dataset import DataLoader, MultiDomainDataLoader
from utils.snips_e2ebert_domain_datautil import load_data, create_vocab, batch_variable, save_to, domain2slot, slot2desp
# from data.snips.generate_slu_emb import slot_list
from utils.conlleval import evaluate
from logger.logger import logger
from torch.nn.utils.rnn import pad_sequence


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        print(self.args)
        
        self.all_dataset = load_data()
        self.tgt_dm = args.tgt_dm
        self.n_sample = args.n_sample
        
        self.train_set = {}
        self.val_set = []
        self.test_set = []
        # split train/dev set.
        for dm, insts in self.all_dataset.items():
            if dm != self.tgt_dm:
                if dm not in self.train_set:
                    self.train_set[dm] = []
                self.train_set[dm].extend(insts)
            else:
                if dm not in self.train_set:
                    self.train_set[dm] = []
                self.train_set[dm].extend(insts[: self.n_sample])
                self.val_set.extend(insts[self.n_sample: 500])
                self.test_set.extend(insts[500:])

        print('train data size:', sum([len(x[1]) for x in self.train_set.items()]))
        print('validate data size:', len(self.val_set))
        print('test data size:', len(self.test_set))

        self.train_loader = MultiDomainDataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.val_set, batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size, shuffle=False)

        self.vocabs = create_vocab(self.all_dataset, args.bert_path)
        save_to(args.vocab_ckpt, self.vocabs)

        print([x for x in self.vocabs["binary"]])
        
        self.slu_model = End2endSLUTagger(
            self.vocabs,
            bert_embed_dim=768,
            num_bound=len(self.vocabs['binary']),
            num_bert_layer=4,
            dropout=args.dropout,
            use_cl=args.cl,
            bert_model_path='bert_model/bert-base-uncased'
        ).to(args.device)
        
        print(self.slu_model)
        total_params = sum([p.numel() for gen in [self.slu_model.parameters()] for p in gen if p.requires_grad])
        print("Training %dM trainable parameters..." % (total_params/1e6))
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 1e-2, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},
            # {'params': [p for n, p in self.model.base_named_params() if p.requires_grad],
            # 'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate}
            {'params': [p for n, p in self.slu_model.base_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-4, 'lr': self.args.lr},
            {'params': [p for n, p in self.slu_model.base_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.lr},
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        max_step = len(self.train_loader) * self.args.epoch
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 10, t_total=max_step)
    

    def train_epoch(self, ep=0):
        self.slu_model.train()
        
        t1 = time.time()
        loss_tr = {"bio_loss": 0., "slu_loss": 0., "cl_loss": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            
            O_tag_idx = self.vocabs['binary'].inst2idx('O')
            # if i == 0:
            #     print(batch_train_data[0].domain)
            loss, dict = self.slu_model(batch.bert_inputs, len(domain2slot[batch_train_data[0].domain]), batch.bio_label, batch.slu_label, O_tag_idx, 
                                        mask=batch.token_mask)
            loss_tr["bio_loss"] = dict["bio_loss"]
            loss_tr["slu_loss"] = dict["slu_loss"]
            loss_tr["cl_loss"] = dict["cl_loss"]
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.slu_model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f, slu_loss: %.4f, cl_loss: %.4f' % \
                (ep, i, (time.time() - t1), loss.item(), loss_tr["bio_loss"], loss_tr["slu_loss"], loss_tr["cl_loss"]))   

        return loss_tr


    def save_states(self, save_path, best_test_metric=None):
        self.slu_model.zero_grad()
        check_point = {'slu_model_state': self.slu_model.state_dict(),
                       'args': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')
        

    def train(self):
        patient = 0
        best_dev_metric, best_test_metric = dict(), dict()
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.train_epoch(ep)

            dev_metric = self.evaluate(self.dev_loader)
            if dev_metric['slu_f'] > best_dev_metric.get('slu_f', 0):
                best_dev_metric = dev_metric
                self.save_states(self.args.model_ckpt, best_dev_metric)
                test_metric = self.evaluate(self.test_loader)
                if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
                   best_test_metric = test_metric
                #    logger.info(f"best test metric saved!, {best_test_metric}")
                #    self.save_states(self.args.model_ckpt, best_test_metric)
                patient = 0
            else:
                patient += 1

            if patient >= self.args.patient:
                print(f"======early stopping with patient:{patient}======")
                break
            logger.info('[Epoch %d] train loss: %s, patient: %d, dev_metric: %s, test_metric: %s' %\
                (ep, train_loss, patient, best_dev_metric, best_test_metric))

        test_metric = self.evaluate(self.test_loader)
        if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
           best_test_metric = test_metric
        #    self.save_states(self.args.model_chkp, best_test_metric)
        logger.info(f'Final Dev Metric: {best_dev_metric}, Final Test Metric: {best_test_metric}')
        return test_metric


    def evaluate(self, test_loader, output_emb=False):
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        # # TODO: bad case analysis
        # wrong_samples = []
        # wrong_pred = []
        # wrong_gold = []
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs, mode='test')
                batch.to_device(self.args.device)
                
                batch_bio_pred, batch_slu_pred = self.slu_model.evaluate(batch.bert_inputs, len(domain2slot[batcher[0].domain]), batch.token_mask)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, (bio_pred, slu_pred) in enumerate(zip(batch_bio_pred, batch_slu_pred)):
                    bio_gold_tags.extend([x.split('-')[0] if x != 'O' else x for x in batcher[j].slu_tags])
                    slu_gold_tags.extend(batcher[j].slu_tags)

                    sent_bio_pred = self.vocabs['binary'].idx2inst(bio_pred[:seq_len[j]].tolist())
                    sent_slu_pred = [slot2desp[domain2slot[batcher[j].domain][x]] for x in slu_pred[:seq_len[j]].tolist()]
                    sent_slu_pred = [x+'-'+y if x != 'O' else 'O' for x, y in zip(sent_bio_pred, sent_slu_pred)]
                    bio_pred_tags.extend(sent_bio_pred)
                    slu_pred_tags.extend(sent_slu_pred)

        #             # TODO: bad case analysis
        #             if sum([x != y for x, y in zip(sent_slu_pred, batcher[j].slu_tags)]) > 0:
        #                 wrong_samples.append(' '.join(batcher[j].tokens))
        #                 wrong_pred.append(' '.join(sent_slu_pred))
        #                 wrong_gold.append(' '.join(batcher[j].slu_tags))
        # with open('data/snips/badcase_rb.txt', 'a+') as f:
        #     for sent, pred, gold in zip(wrong_samples, wrong_pred, wrong_gold):
        #         f.write(sent + '\n')
        #         f.write(pred + '\n')
        #         f.write(gold + '\n\n')
                    
        assert len(slu_pred_tags) == len(slu_gold_tags)
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


# train
# nohup python slu_e2e_bert_domain_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 20 --dropout 0.3 --cl --model_ckpt ckpt/end2end_cl/bert_domain_vae_atp0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_vae_atp0_vocab.ckpt &> training_log/end2end_cl/bert_domain_vae_atp0.log &
# nohup python slu_e2e_bert_domain_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 20 --dropout 0.3 --cl --model_ckpt ckpt/end2end_cl/bert_domain_vae_br0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_vae_br0_vocab.ckpt &> training_log/end2end_cl/bert_domain_vae_br0.log &
# nohup python slu_e2e_bert_domain_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 20 --dropout 0.3 --cl --model_ckpt ckpt/end2end_cl/bert_domain_vae_gw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_vae_gw0_vocab.ckpt &> training_log/end2end_cl/bert_domain_vae_gw0.log &
# nohup python slu_e2e_bert_domain_train.py --cuda 3 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 20 --dropout 0.3 --cl --model_ckpt ckpt/end2end_cl/bert_domain_vae_pm0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_vae_pm0_vocab.ckpt &> training_log/end2end_cl/bert_domain_vae_pm0.log &
# nohup python slu_e2e_bert_domain_train.py --cuda 4 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 20 --dropout 0.3 --cl --model_ckpt ckpt/end2end_cl/bert_domain_vae_rb0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_vae_rb0_vocab.ckpt &> training_log/end2end_cl/bert_domain_vae_rb0.log &
# nohup python slu_e2e_bert_domain_train.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 20 --dropout 0.3 --cl --model_ckpt ckpt/end2end_cl/bert_domain_vae_scw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_vae_scw0_vocab.ckpt &> training_log/end2end_cl/bert_domain_vae_scw0.log &
# nohup python slu_e2e_bert_domain_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 20 --dropout 0.3 --cl --model_ckpt ckpt/end2end_cl/bert_domain_vae_sse0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_vae_sse0_vocab.ckpt &> training_log/end2end_cl/bert_domain_vae_sse0.log &

# test
# nohup python slu_end2end_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 20 --model_ckpt testlog/test_atp0.ckpt &> testlog/test_orthlr1e-4_temp0.5_atp0.log &
# nohup python slu_e2e_bert_domain_train.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 20 --dropout 0.3 --cl --model_ckpt testlog/test.ckpt --vocab_ckpt ckpt/vocab/test_vocab.ckpt &> testlog/bert_domain_unseen_test1_br0.log &
# nohup python slu_end2end_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 20 --model_ckpt testlog/test_gw0.ckpt &> testlog/test_orthlr1e-4_temp0.5_gw0.log &
# nohup python slu_end2end_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 20 --model_ckpt testlog/test_pm0.ckpt &> testlog/test_orthlr1e-4_temp0.5_pm0.log &
# nohup python slu_end2end_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 20 --model_ckpt testlog/test_rb0.ckpt &> testlog/test_orthlr1e-4_temp0.5_rb0.log &
# nohup python slu_e2e_bert_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 20 --model_ckpt testlog/test_scw0.ckpt &> testlog/test_bert_orth_scw0.log &
# nohup python slu_e2e_bert_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 20 --model_ckpt testlog/test_sse0.ckpt &> testlog/test_bert_woorth_sse0.log &