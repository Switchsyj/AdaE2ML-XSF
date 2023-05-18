import time
import torch
import torch.nn as nn
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.ner_model import End2endSLUTagger
from config.slu_conf import args_config
from utils.dataset import DataLoader
from utils.ner_datautil import load_data, create_vocab, batch_variable, save_to
from utils.conlleval import evaluate
import copy
from logger.logger import logger


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        print(self.args)
        
        self.n_sample = args.n_sample
        
        self.train_set = load_data('conll', 'data/ner/conll2003/train.txt')
        self.val_set = load_data('conll', 'data/ner/conll2003/dev.txt')
        self.test_set = load_data('tech', 'data/ner/tech/tech_test.txt')
        # few-shot setting
        if self.args.n_sample > 0:
            self.train_set.extend(self.test_set[:self.args.n_sample])
            self.test_set = self.test_set[self.args.n_sample:]

        self.all_datasets = self.train_set + self.val_set + self.test_set
 
        print('train data size:', len(self.train_set))
        print('validate data size:', len(self.val_set))
        print('test data size:', len(self.test_set))

        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.val_set, batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size, shuffle=False)

        self.vocabs = create_vocab(self.all_datasets, 'bert_model/bert-base-uncased')
        print(len(self.vocabs['slot']))
        # save_to(args.vocab_chkp, self.vocabs)
    
        self.slu_model = End2endSLUTagger(
            self.vocabs,
            bert_embed_dim=768,
            num_label=len(self.vocabs['slot']),
            num_bert_layer=4,
            dropout=args.dropout,
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
             'weight_decay': 0.0, 'lr': self.args.lr}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        max_step = len(self.train_loader) * self.args.epoch
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 10, t_total=max_step)
    

    def train_epoch(self, ep=0):
        self.slu_model.train()
        
        t1 = time.time()
        loss_tr = {"tag_loss": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            loss, dict = self.slu_model(batch.bert_inputs, batch.slu_label, mask=batch.token_mask)
            loss_tr["tag_loss"] = dict["tag_loss"]
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.slu_model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.4fs, loss: %.4f' % (ep, i, (time.time() - t1), loss.item()))
                
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
                # self.save_states(self.args.model_ckpt, best_dev_metric)
                best_test_metric = self.evaluate(self.test_loader)
                patient = 0
            else:
                patient += 1

            if patient >= self.args.patient:
                print(f"======early stopping with patient:{patient}======")
                break
            logger.info('[Epoch %d] train loss: %s, patient: %d, dev_metric: %s, test_metric: %s' %\
                (ep, train_loss, patient, best_dev_metric, best_test_metric))

        logger.info(f'Final Dev Metric: {best_dev_metric}, Final Test Metric: {best_test_metric}')
        return best_test_metric


    def evaluate(self, test_loader):
        self.slu_model.eval()

        slu_pred_tags = []
        slu_gold_tags = []
        with torch.no_grad():
            for _, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs, mode='test')
                batch.to_device(self.args.device)
                
                pred_labels = self.slu_model.evaluate(batch.bert_inputs, batch.token_mask)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, pred_label in enumerate(pred_labels):
                    slu_gold_tags.extend(batcher[j].ner_tags)
                    sent_slu_pred = self.vocabs['slot'].idx2inst(pred_label[:seq_len[j]].tolist())
                    slu_pred_tags.extend(sent_slu_pred)
                    
        assert len(slu_pred_tags) == len(slu_gold_tags)
        slu_p, slu_r, slu_f = evaluate(slu_gold_tags, slu_pred_tags, verbose=False) 
        return {"slu_p": slu_p, "slu_r": slu_r, "slu_f": slu_f}


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

# nohup python ner_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.3 --model_ckpt testlog/test_tech.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/backbone_ner_b-crf.log &