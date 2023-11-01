import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.e2e_bert_f2tagger import End2endSLUTagger
from config.xslu_conf import args_config
from utils.dataset import DataLoader, MultiDomainDataLoader
from utils.xDataset_datautil import load_data, create_vocab, batch_variable, save_to, slot2desp_textbook
from utils.conlleval import evaluate
from logger.logger import logger
from copy import deepcopy
import json


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        print(self.args)
        
        self.all_dataset = load_data(args.src, args.tgt)
        
        self.train_set = self.all_dataset['train']
        self.val_set = self.all_dataset['dev']
        self.test_set = self.all_dataset['test']

        print('train data size:', sum([len(x[1]) for x in self.train_set.items()]))
        print('validate data size:', sum([len(x[1]) for x in self.val_set.items()]))
        print('test data size:', sum([len(x[1]) for x in self.test_set.items()]))
        
        self.train_loader = MultiDomainDataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.dev_loader = MultiDomainDataLoader(self.val_set, batch_size=self.args.val_batch_size, shuffle=False)  
        self.test_loader = MultiDomainDataLoader(self.test_set, batch_size=self.args.test_batch_size, shuffle=False)
 
        self.vocabs = create_vocab(self.all_dataset, args.bert_path)

        print([x for x in self.vocabs["binary"]])
        self.domain2slot = dict()
        with open(f"data/merge_dataset/{args.src}/labels.json", 'r') as f:
            _temp = json.load(f)
            for k, v in _temp.items():
                self.domain2slot[k] = ['[PAD]'] + [x for x in v if x != 'O']
            del _temp
        print("======domain2slot for source domain: ======\n", self.domain2slot)
        print(f"****** average slots number: {np.mean(np.array([len(x) for _, x in self.domain2slot.items()]))} ******")
        with open(f"data/merge_dataset/{args.tgt}/labels.json", 'r') as f:
            _temp = json.load(f)
            for k, v in _temp.items():
                if self.domain2slot.get(k, None) == None:
                    self.domain2slot[k] = ['[PAD]'] + [x for x in v if x != 'O']
                else:
                    self.domain2slot[k] = ['[PAD]'] + list(set(self.domain2slot[k][1:] + [x for x in v if x != 'O']))
            del _temp
        print("======domain2slot for source and target domain: ======\n", self.domain2slot)

        overall_slots = []
        for _, v in self.domain2slot.items():
            overall_slots.extend(v)
        overall_slots = list(set(overall_slots))
        
        self.slot2desp = {'[PAD]': '[PAD]'}
        for slot in overall_slots:
            if slot in slot2desp_textbook:
                self.slot2desp[slot] = slot2desp_textbook[slot]
            else:
                self.slot2desp[slot] = slot.replace('.', ' ').replace('_', ' ').replace('-', ' ')
        print(f"******Loading {len(overall_slots)} slots for training and testing ******")
        
        self.slu_model = End2endSLUTagger(
            self.vocabs,
            bert_embed_dim=768,
            num_bound=len(self.vocabs['binary']),
            dropout=args.dropout,
            use_cl=args.cl,
            cl_type=args.cl_type,
            bert_model_path='bert_model/bert-base-uncased',
            cl_temperature=args.cl_temperature,
            alpha=args.alpha,
            beta=args.beta,
            ).to(args.device)
        
        print(self.slu_model)
        total_params = sum([p.numel() for gen in [self.slu_model.parameters()] for p in gen if p.requires_grad])
        print("Training %dM trainable parameters..." % (total_params/1e6))
                
        no_decay = ['bias', 'LayerNorm.weight']
        labelspace_param = ['adapter']
        
        optimizer_parameters = [
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 1e-2, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},
            
            {'params': [p for n, p in self.slu_model.base_named_params() if not any(nd in n for nd in no_decay) and any(lp in n for lp in labelspace_param)],
             'weight_decay': 1e-4, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.slu_model.base_named_params() if any(nd in n for nd in no_decay) and any(lp in n for lp in labelspace_param)],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},
            
            {'params': [p for n, p in self.slu_model.base_named_params() if not any(nd in n for nd in no_decay) and not any(lp in n for lp in labelspace_param)],
             'weight_decay': 1e-4, 'lr': self.args.lr},
            {'params': [p for n, p in self.slu_model.base_named_params() if any(nd in n for nd in no_decay) and not any(lp in n for lp in labelspace_param)],
             'weight_decay': 0.0, 'lr': self.args.lr},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        max_step = len(self.train_loader) * self.args.epoch
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 10, t_total=max_step)

    def train_epoch(self, ep=0):
        self.slu_model.train()
        t1 = time.time()
        loss_tr = {"bio_loss": 0., "slu_loss": 0., "cl_loss": 0., "orth_loss": 0., "utter": 0., "label": 0., "vae_kl": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            batch = batch_variable(batch_train_data, self.vocabs, self.slot2desp, self.domain2slot, mode='train')
            batch.to_device(self.args.device)
            
            O_tag_idx = self.vocabs['binary'].inst2idx('O')
            loss, dict = self.slu_model(batch.bert_inputs,
                                        len(self.domain2slot[batch_train_data[0].domain]),
                                        batch.bio_label, batch.slu_label, O_tag_idx, 
                                        mask=batch.token_mask,
                                        train='utter',
                                        )
            loss_tr["bio_loss"] = dict["bio_loss"]
            loss_tr["slu_loss"] = dict["slu_loss"]
            loss_tr["cl_loss"] = dict["cl_loss"]
            logger.info('[Epoch %d] [lr: %.4f] Utter Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f, slu_loss: %.4f, cl_loss: %.4f' % \
                (ep, self.scheduler.get_last_lr()[-1], i, (time.time() - t1), loss.item(), loss_tr["bio_loss"], loss_tr["slu_loss"], loss_tr["cl_loss"]))
            
            loss.backward()
            self.optimizer.step()
            self.slu_model.zero_grad()
            loss, dict = self.slu_model(batch.bert_inputs,
                            len(self.domain2slot[batch_train_data[0].domain]),
                            batch.bio_label, batch.slu_label, O_tag_idx, 
                            mask=batch.token_mask,
                            train='label',
                            )
            loss_tr["bio_loss"] = dict["bio_loss"]
            loss_tr["slu_loss"] = dict["slu_loss"]
            loss_tr["cl_loss"] = dict["cl_loss"]
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.slu_model.zero_grad()

            logger.info('[Epoch %d] [lr: %.4f] Label Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f, slu_loss: %.4f, cl_loss: %.4f' % \
                (ep, self.scheduler.get_last_lr()[-1], i, (time.time() - t1), loss.item(), loss_tr["bio_loss"], loss_tr["slu_loss"], loss_tr["cl_loss"]))
        return loss_tr


    def save_states(self, save_path, best_metric=None):
        self.slu_model.zero_grad()
        check_point = {'slu_model_state': self.slu_model.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'best_prf': best_metric,
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
                inf_st_time = time.time()
                best_test_metric = self.evaluate(self.test_loader)
                print(f"======inference time: {time.time()-inf_st_time}======")
                patient = 0
            else:
                patient += 1

            if patient >= self.args.patient:
                print(f"======early stopping with patient:{patient}======")
                break
            logger.info('[Epoch %d] train loss: %s, patient: %d, dev_metric: %s, best_dev_metric: %s, best_test_metric: %s' %\
                (ep, train_loss, patient, dev_metric, best_dev_metric, best_test_metric))

        logger.info(f'Training finished, best dev metric: {best_dev_metric}, best test metric: {best_test_metric}')
        
        return best_test_metric

    def evaluate(self, test_loader, output_emb=False):
        """
        Dev set evaluation
        """
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs, self.slot2desp, self.domain2slot, mode='test')
                batch.to_device(self.args.device)
                
                batch_bio_pred, batch_slu_pred = self.slu_model.evaluate(batch.bert_inputs, 
                                                                         len(self.domain2slot[batcher[0].domain]), 
                                                                         batch.token_mask)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, (bio_pred, slu_pred) in enumerate(zip(batch_bio_pred, batch_slu_pred)):
                    bio_gold_tags.extend([x.split('-')[0] if x != 'O' else x for x in batcher[j].slu_tags])
                    slu_gold_tags.extend([x for x in batcher[j].slu_tags])

                    sent_bio_pred = self.vocabs['binary'].idx2inst(bio_pred[:seq_len[j]].tolist())
                    sent_slu_pred = []
                    prev = None
                    for _bio, _slu in zip(sent_bio_pred, slu_pred[:seq_len[j]].tolist()):
                        slotname = self.domain2slot[batcher[j].domain][_slu]
                        if slotname == 'O':
                            sent_slu_pred.append('O')
                            continue
                        if _bio == 'B':
                            prev = slotname
                            sent_slu_pred.append('B-' + prev)
                        elif _bio == 'I':
                            if prev is None:
                                sent_slu_pred.append('B-' + slotname)
                                prev = slotname
                            else:
                                sent_slu_pred.append('I-' + prev)
                        else:
                            sent_slu_pred.append('O')
                            prev = None
                    bio_pred_tags.extend(sent_bio_pred)
                    slu_pred_tags.extend(sent_slu_pred)
                    
        assert len(slu_pred_tags) == len(slu_gold_tags)
        bio_p, bio_r, bio_f = evaluate([x + '-1' if x != 'O' else x for x in bio_gold_tags], 
                                       [x + '-1' if x != 'O' else x for x in bio_pred_tags], verbose=False)
        slu_p, slu_r, slu_f = evaluate(slu_gold_tags, slu_pred_tags, verbose=False)
        return {"bio_p": bio_p, "bio_r": bio_r, "bio_f": bio_f, \
                "slu_p": slu_p, "slu_r": slu_r, "slu_f": slu_f}

    def reset_parameters(self, model_ckpt_path):
        """
        load model from best-dev model dict.
        """
        print(f'Loading the previous model states from {model_ckpt_path}')
        ckpt = torch.load(model_ckpt_path)
        
        if not self.args.cl:
            self.slu_model.load_state_dict(deepcopy({k: v for k, v in ckpt['slu_model_state'].items() if 'tokencl' not in k}))
        else:
            self.slu_model.load_state_dict(deepcopy({k: v for k, v in ckpt['slu_model_state'].items()}))
        
        self.slu_model.zero_grad()
        self.slu_model.train()
        
        print('Previous best dev prf result is: %s' % ckpt['best_prf'])


def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    
    args = args_config()
    
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')
    
    random_seed = [1314, 1315, 1316, 1317, 1318]
    final_res = {'p': [], 'r': [], 'f': []}
    for seed in random_seed:
        set_seeds(seed)
        print(f"set seed: {seed}")
        trainer = Trainer(args, data_config=None)
        prf = trainer.train()
        
        final_res['p'].append(prf['slu_p'])
        final_res['r'].append(prf['slu_r'])
        final_res['f'].append(prf['slu_f'])
        break
    final_res['p'] = np.array(final_res['p'])
    final_res['r'] = np.array(final_res['r'])
    final_res['f'] = np.array(final_res['f'])
    print(f"avg result: p: {final_res['p'].mean()}+-{final_res['p'].std()}, r: {final_res['r'].mean()}+-{final_res['r'].std()}, f: {final_res['f'].mean()}+-{final_res['f'].std()}")
