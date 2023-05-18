import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.cascade_tagger import CascadeTagger
from config.slu_conf import args_config
from utils.dataset import DataLoader
from utils.cascade_datautil import load_data, create_vocab, batch_variable, save_to
import torch.nn.utils as nn_utils
from logger.logger import logger
from utils.conlleval import evaluate


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # self.data_config = data_config
        self.train_set = load_data('data/snips/AddToPlaylist/AddToPlaylist.txt') + \
            load_data('data/snips/GetWeather/GetWeather.txt') + \
            load_data('data/snips/PlayMusic/PlayMusic.txt') + \
            load_data('data/snips/RateBook/RateBook.txt') + \
            load_data('data/snips/SearchCreativeWork/SearchCreativeWork.txt') + \
            load_data('data/snips/SearchScreeningEvent/SearchScreeningEvent.txt')
        print('train data size:', len(self.train_set))
        self.dev_set = load_data('data/snips/BookRestaurant/BookRestaurant.txt')[:500]
        self.test_set = load_data('data/snips/BookRestaurant/BookRestaurant.txt')[500:]
        print('train data size:', len(self.train_set))
        print('dev data size:', len(self.dev_set))
        print('test data size:', len(self.test_set))
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.test_batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size)

        self.vocabs = create_vocab(self.train_set+self.dev_set+self.test_set, 'bert_model/bert-base-uncased', embed_file=None)
        # save_to(args.vocab_chkp, self.vocabs)
        print('slot vocab length', len(self.vocabs['type']))
        print('bio vocab length', len(self.vocabs['bound']))
        print(self.vocabs['bound']._inst2idx.keys(), self.vocabs['type']._inst2idx.keys())

        self.model = CascadeTagger(
            self.vocabs,
            bert_embed_dim=768,
            hidden_size=0,
            num_type=len(self.vocabs['type']),
            num_bound=len(self.vocabs['bound']),
            num_bert_layer=4,
            dropout=args.dropout,
            bert_model_path='bert_model/bert-base-uncased'
        ).to(args.device)
        
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Training %d M trainable parameters..." % (total_params/1e6))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': self.args.weight_decay, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},
            # {'params': [p for n, p in self.model.base_named_params() if p.requires_grad],
            # 'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate}
            {'params': [p for n, p in self.model.base_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lr},
            {'params': [p for n, p in self.model.base_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.lr}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_bert_parameters, lr=self.args.bert_lr)
        max_step = len(self.train_loader) * self.args.epoch
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 10, t_total=max_step)

    def train_epoch(self, ep=1):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        for i, batch_train_data in enumerate(self.train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            O_idx = self.vocabs['bound'].inst2idx('O')
            loss = self.model(batch.bert_inp, batch.bound_ids, batch.type_ids, O_idx, batch.mask)
            loss_val = loss.data.item()
            train_loss += (loss_val / len(self.train_loader))
            loss.backward()
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=self.args.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, train loss: %.3f' % (ep, i, (time.time() - t1), loss_val))
        return train_loss

    def train(self):
        best_dev_metric, best_test_metric = dict(), dict()
        patient = 0
        for ep in range(1, 1 + self.args.epoch):
            train_loss = self.train_epoch(ep)
            dev_metric = self.evaluate(self.dev_loader)
            if dev_metric['f'] > best_dev_metric.get('f', 0):
                best_dev_metric = dev_metric
                test_metric = self.evaluate(self.test_loader)
                if test_metric['f'] > best_test_metric.get('f', 0):
                    best_test_metric = test_metric
                    # self.save_states(self.args.model_chkp, best_test_metric)
                patient = 0
            else:
                patient += 1

            logger.info('[Epoch %d] train loss: %.4f, patient: %d, dev_metric: %s, test_metric: %s' % (
                ep, train_loss, patient, best_dev_metric, best_test_metric))

            if patient >= self.args.patient:  # early stopping
                break

        test_metric = self.evaluate(self.test_loader)
        if test_metric['f'] > best_test_metric.get('f', 0):
            best_test_metric = test_metric
            # self.save_states(self.args.model_chkp, best_test_metric)

        logger.info('Final Dev Metric: %s, Test Metric: %s' % (best_dev_metric, best_test_metric))
        return best_test_metric

    def evaluate(self, test_loader):
        bound_vocab = self.vocabs['bound']
        type_vocab = self.vocabs['type']
        test_pred_tags, test_gold_tags = [], []
        self.model.eval()
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                pred_bound, pred_type = self.model.inference(batch.bert_inp, batch.mask)
                seq_lens = batch.mask.sum(dim=1).tolist()
                for j, l in enumerate(seq_lens):
                    pred_bound_tags = bound_vocab.idx2inst(pred_bound[j][1:l].tolist())
                    pred_type_tags = type_vocab.idx2inst(pred_type[j][1:l].tolist())
                    pred_tags = [b+'-'+t if b != 'O' else 'O' for b, t in zip(pred_bound_tags, pred_type_tags)]
                    gold_tags = batcher[j].slu_tags
                    test_pred_tags.extend(pred_tags)
                    test_gold_tags.extend(gold_tags)
                    assert len(test_gold_tags) == len(test_pred_tags)

        p, r, f = evaluate(test_gold_tags, test_pred_tags, verbose=False)
        return dict(p=p, r=r, f=f)

    def save_states(self, save_path, best_test_metric=None):
        check_point = {'best_prf': best_test_metric,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'args_settings': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')

    def restore_states(self, load_path):
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.args = ckpt['args_settings']
        logger.info('Loading the previous model states ...')
        print('Previous best prf result is: %s' % ckpt['best_prf'])


def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')

    # data_path = data_config('./config/data_path.json')

    random_seeds = [1357, 2789, 3391, 4553, 5919]
    final_res = {'p': [], 'r': [], 'f': []}
    for seed in random_seeds:
        set_seeds(seed)
        trainer = Trainer(args)
        prf = trainer.train()
        break

    # logger.info('Final Result: %s' % final_res)
    # final_p = sum(final_res['p']) / len(final_res['p'])
    # final_r = sum(final_res['r']) / len(final_res['r'])
    # final_f = sum(final_res['f']) / len(final_res['f'])
    # logger.info('Final P: %.4f, R: %.4f, F: %.4f' % (final_p, final_r, final_f))