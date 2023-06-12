import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.e2e2step_tagger import End2endSLUTagger
from config.slu_conf import args_config
from utils.dataset import DataLoader, MultiDomainDataLoader
from utils.slu_2step_datautil import load_data, create_vocab, batch_variable, batch_ptr_variable, save_to, domain2slot, slot2desp, domain2unseen
from utils.conlleval import evaluate
from logger.logger import logger
from copy import deepcopy as dc

# TODO: wandb
# import wandb
# import yaml


class PretrainDataLoader(object):
    def __init__(self, dataset: dict, batch_size=1, shuffle=False, collate_fn=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.dataset = []
        
        if isinstance(dataset, dict):
            for k, v in dataset.items():
                dm_inst = []
                for _inst in v:
                    for _tag in domain2slot[_inst.domain][1:]:
                        tmp_inst = dc(_inst)
                        tmp_inst.tokens = [_tag] + tmp_inst.tokens
                        dm_inst.append(tmp_inst) 
                self.dataset.extend(dm_inst)
        elif isinstance(dataset, list):
            dm_inst = []
            for _inst in dataset:
                for _tag in domain2slot[_inst.domain][1:]:
                    tmp_inst = dc(_inst)
                    tmp_inst.tokens = [_tag] + tmp_inst.tokens
                    dm_inst.append(tmp_inst) 
            self.dataset.extend(dm_inst)
        else:
            raise NotImplementedError

    def __iter__(self):
        n = len(self.dataset)
        # 打乱顺序
        if self.shuffle:
            idxs = np.random.permutation(n)
        else:
            idxs = range(n)

        batch = []
        # batch(Instance1, Instance2, ...)
        for idx in idxs:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch
                batch = []

        # 剩余的batch
        if len(batch) > 0:
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                yield batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


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
        # self.ptr_train_loader = PretrainDataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        # self.ptr_dev_loader = PretrainDataLoader(self.val_set, batch_size=self.args.batch_size, shuffle=False)
        self.ptr_train_loader = MultiDomainDataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.ptr_dev_loader = DataLoader(self.val_set, batch_size=self.args.batch_size, shuffle=False)
        
        self.dev_loader = DataLoader(self.val_set, batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size, shuffle=False)
 
        self.vocabs = create_vocab(self.all_dataset, args.bert_path)
        # TODO: add special token [ENT]
        # print(self.vocabs['bert'].bert2id(['[ENT]']))
        # self.vocabs['bert'].bert_tokenizer.add_tokens(['[ENT]'], special_tokens=True)   
        # print(self.vocabs['bert'].bert2id(['[ENT]']))
        # save_to(args.vocab_ckpt, self.vocabs)

        print([x for x in self.vocabs["binary"]])
        
        self.slu_model = End2endSLUTagger(
            self.vocabs,
            bert_embed_dim=768,
            num_bound=len(self.vocabs['binary']),
            num_bert_layer=4,
            dropout=args.dropout,
            use_cl=args.cl,
            cl_type=args.cl_type,
            bert_model_path='bert_model/bert-base-uncased',
            cl_temperature=args.cl_temperature,
            ).to(args.device)
        
        # TODO: add special token [ENT]
        # self.slu_model.bert.bert.resize_token_embeddings(len(self.vocabs['bert'].bert_tokenizer))
        
        print(self.slu_model)
        total_params = sum([p.numel() for gen in [self.slu_model.parameters()] for p in gen if p.requires_grad])
        print("Training %dM trainable parameters..." % (total_params/1e6))
    
    def pretrain_epoch(self, ep=0):
        self.slu_model.train()
        t1 = time.time()
        loss_ptr = {"bio_loss": 0.}
        for i, batch_train_data in enumerate(self.ptr_train_loader):
            # mask_p = self.get_mask_p(ep-1)
            batch = batch_ptr_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            
            O_tag_idx = self.vocabs['binary'].inst2idx('O')
            loss, dict = self.slu_model.pretrain(batch.bert_inputs,
                                                batch.bio_label, 
                                                O_tag_idx, 
                                                mask=batch.token_mask,
                                                )
            loss_ptr["bio_loss"] = dict["bio_loss"]
            logger.info('[Epoch %d] [lr: %.4f] Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f' % \
                (ep, self.scheduler.get_last_lr()[-1], i, (time.time() - t1), loss.item(), loss_ptr["bio_loss"]))
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.slu_model.zero_grad()
            
            # TODO: wandb
            # wandb.log({
            #     "slu_loss": loss.item(),
            #     "lr": self.scheduler.get_last_lr()[-1]
            # })
            
        return loss_ptr
        
    def train_epoch(self, ep=0):
        self.slu_model.train()
        t1 = time.time()
        loss_tr = {"bio_loss": 0., "slu_loss": 0., "cl_loss": 0., "orth_loss": 0., "utter": 0., "label": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            # mask_p = self.get_mask_p(ep-1)
            batch = batch_variable(batch_train_data, self.vocabs, self.args.tgt_dm, mode='train')
            batch.to_device(self.args.device)
            
            O_tag_idx = self.vocabs['binary'].inst2idx('O')
            loss, dict = self.slu_model(batch.bert_inputs,
                                        len(domain2slot[batch_train_data[0].domain]),
                                        batch.bio_label, batch.slu_label, O_tag_idx, 
                                        mask=batch.token_mask,
                                        train='label',
                                        )
            loss_tr["bio_loss"] = dict["bio_loss"]
            loss_tr["slu_loss"] = dict["slu_loss"]
            loss_tr["cl_loss"] = dict["cl_loss"]
            loss_tr["orth_loss"] = dict["orth_loss"]
            # loss_tr["utter"] = dict["utter"]
            # loss_tr["label"] = dict["label"]
            logger.info('[Epoch %d] [lr: %.4f] Utter Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f, slu_loss: %.4f, cl_loss: %.4f, orth_loss: %.4f' % \
                (ep, self.scheduler.get_last_lr()[-1], i, (time.time() - t1), loss.item(), loss_tr["bio_loss"], loss_tr["slu_loss"], loss_tr["cl_loss"], loss_tr["orth_loss"]))
            
            loss.backward()
            self.optimizer.step()
            self.slu_model.zero_grad()
            loss, dict = self.slu_model(batch.bert_inputs,
                            len(domain2slot[batch_train_data[0].domain]),
                            batch.bio_label, batch.slu_label, O_tag_idx, 
                            mask=batch.token_mask,
                            train='utter',
                            )
            loss_tr["bio_loss"] = dict["bio_loss"]
            loss_tr["slu_loss"] = dict["slu_loss"]
            loss_tr["cl_loss"] = dict["cl_loss"]
            loss_tr["orth_loss"] = dict["orth_loss"]
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.slu_model.zero_grad()
            
            # TODO: wandb
            # wandb.log({
            #     "slu_loss": loss.item(),
            #     "lr": self.scheduler.get_last_lr()[-1]
            # })

            logger.info('[Epoch %d] [lr: %.4f] Label Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f, slu_loss: %.4f, cl_loss: %.4f, orth_loss: %.4f' % \
                (ep, self.scheduler.get_last_lr()[-1], i, (time.time() - t1), loss.item(), loss_tr["bio_loss"], loss_tr["slu_loss"], loss_tr["cl_loss"], loss_tr["orth_loss"]))
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
        # TODO: wandb
        # run = wandb.init(project='backbone-cl-sweep', config = {}, entity='switchsyj')
        # wconfig = wandb.config
        # run.name = f"ep: {wconfig['ep']}, step: {wconfig['step']}, ceil: {wconfig['ceil']}"
        # self.ep = 2
        # self.step = 0.02
        # self.ceil = 0.1
        
        patient = 0
        best_dev_metric, best_test_metric = dict(), dict()
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 1e-2, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},
            
            {'params': [p for n, p in self.slu_model.base_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-4, 'lr': self.args.lr},
            {'params': [p for n, p in self.slu_model.base_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.lr},
            
        ]
        
        print("====== Pretraining ======")
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        max_step = len(self.ptr_train_loader) * 1
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 15, t_total=max_step)
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.pretrain_epoch(ep)
            dev_metric = self.evaluate_ptr(self.ptr_dev_loader)
            if dev_metric['bio_f'] > best_dev_metric.get('bio_f', 0):
                best_dev_metric = dev_metric
                self.save_states(self.args.model_ckpt, best_dev_metric)
                patient = 0
            else:
                patient += 1
            logger.info('[Epoch %d] train loss: %s, patient: %d, dev_metric_p: %s, dev_metric_r: %s, dev_metric_f: %s' %\
                (ep, train_loss, patient, dev_metric['bio_p'], dev_metric['bio_r'], dev_metric['bio_f']))
            if patient >= self.args.patient:
                print(f"======pretraining early stopping with patient:{patient}======")
                break
        
        print("====== Fine tuning ======")
        # reload best BIO parameters
        self.reset_parameters(self.args.model_ckpt)
        optimizer_parameters = [
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 1e-2, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.slu_model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},
            
            {'params': [p for n, p in self.slu_model.base_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-4, 'lr': self.args.lr},
            {'params': [p for n, p in self.slu_model.base_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.lr},
            
        ]
        patient = 0
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        max_step = len(self.train_loader) * self.args.epoch
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 15, t_total=max_step)
        # freeze CRF parameters
        for name, param in self.slu_model.named_parameters():
            if 'crf' in name or 'lstm' in name:
                param.requires_grad_(False)
        
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.train_epoch(ep)

            dev_metric = self.evaluate(self.dev_loader)
            test_metric = self.evaluate(self.test_loader)
            if dev_metric['slu_f'] > best_dev_metric.get('slu_f', 0):
                best_dev_metric = dev_metric
                # self.save_states(self.args.model_ckpt, best_dev_metric)
                best_test_metric = self.evaluate(self.test_loader)
                
                # if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
                #    best_test_metric = test_metric
                #    logger.info(f"best test metric saved!, {best_test_metric}")
                #    self.save_states(self.args.model_ckpt, best_test_metric)
                patient = 0
            else:
                patient += 1
            
            # TODO: wandb
            # wandb.log({"dev_metric_f": dev_metric['slu_f'],
            #            "test_metric_f": test_metric['slu_f']
            #            })

            if patient >= self.args.patient:
                print(f"======early stopping with patient:{patient}======")
                break
            logger.info('[Epoch %d] train loss: %s, patient: %d, dev_metric: %s, best_dev_metric: %s, best_test_metric: %s' %\
                (ep, train_loss, patient, dev_metric, best_dev_metric, best_test_metric))

        # test_metric = self.evaluate(self.test_loader)
        # if test_metric['slu_f'] > best_test_metric.get('slu_f', 0):
        #     best_test_metric = test_metric
        #    self.save_states(self.args.model_chkp, best_test_metric)
        
        # TODO: wandb
        # wandb.log({"best_dev_metric_f": best_dev_metric['slu_f'],
        #            "best_test_metric_f": best_test_metric['slu_f']})

        logger.info(f'Training finished, best dev metric: {best_dev_metric}, best test metric: {best_test_metric}')
        
        return best_test_metric

    def evaluate_ptr(self, ptr_test_loader, output_emb=False):
        """
        Dev set evaluation
        """
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        with torch.no_grad():
            for i, batcher in enumerate(ptr_test_loader):
                batch = batch_ptr_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                
                batch_bio_pred = self.slu_model.evaluate_ptr(batch.bert_inputs, batch.token_mask)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, bio_pred in enumerate(batch_bio_pred):
                    # bio_gold_tags.extend(['O', 'O'] + [x.split('-')[0] if (x != 'O' and x.split('-')[1] == slot2desp[batcher[j].tokens[0]]) else 'O' for x in batcher[j].slu_tags])
                    bio_gold_tags.extend([x.split('-')[0] if x != 'O' else 'O' for x in batcher[j].slu_tags])
                    sent_bio_pred = self.vocabs['binary'].idx2inst(bio_pred[:seq_len[j]].tolist())
                    bio_pred_tags.extend(sent_bio_pred)
                    
        bio_p, bio_r, bio_f = evaluate([x + '-1' if x != 'O' else x for x in bio_gold_tags], 
                                       [x + '-1' if x != 'O' else x for x in bio_pred_tags], verbose=False)
        return {"bio_p": bio_p, "bio_r": bio_r, "bio_f": bio_f}
    
    def evaluate(self, test_loader, output_emb=False):
        """
        Dev set evaluation
        """
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        desp2slot = {v: k for k, v in slot2desp.items()}
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs, self.args.tgt_dm, mode='test')
                batch.to_device(self.args.device)
                
                batch_bio_pred, batch_slu_pred = self.slu_model.evaluate(batch.bert_inputs, len(domain2slot[batcher[0].domain]), batch.token_mask)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, (bio_pred, slu_pred) in enumerate(zip(batch_bio_pred, batch_slu_pred)):
                    bio_gold_tags.extend([x.split('-')[0] if x != 'O' else x for x in batcher[j].slu_tags])
                    slu_gold_tags.extend([x.split('-')[0] + '-' + desp2slot[x.split('-')[1]] if x != 'O' else x for x in batcher[j].slu_tags])

                    sent_bio_pred = self.vocabs['binary'].idx2inst(bio_pred[:seq_len[j]].tolist())
                    # sent_slu_pred = [domain2slot[batcher[j].domain][x] for x in slu_pred[:seq_len[j]].tolist()]
                    # sent_slu_pred = [x+'-'+y if x != 'O' and y != '[PAD]' else 'O' for x, y in zip(sent_bio_pred, sent_slu_pred)]
                    sent_slu_pred = []
                    prev = None
                    for _bio, _slu in zip(sent_bio_pred, slu_pred[:seq_len[j]].tolist()):
                        slotname = domain2slot[batcher[j].domain][_slu]
                        if slotname == '[PAD]':
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
            self.slu_model.load_state_dict(dc({k: v for k, v in ckpt['slu_model_state'].items() if 'tokencl' not in k}))
        else:
            self.slu_model.load_state_dict(dc({k: v for k, v in ckpt['slu_model_state'].items()}))
        
        self.slu_model.zero_grad()
        self.slu_model.train()
        
        print('Previous best dev prf result is: %s' % ckpt['best_prf'])
    
    def test_time_training(self, test_loader):
        """
        Test time training
        """
        self.ttt = TTT(self.vocabs, self.args.model_ckpt, self.args)
        # self.ttt = TTDistill(self.vocabs, self.args.model_ckpt, self.args)
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        desp2slot = {v: k for k, v in slot2desp.items()}
        for i, batcher in enumerate(test_loader):
            batch = batch_variable(batcher, self.vocabs, self.args.tgt_dm, mode='test')
            batch.to_device(self.args.device)
            
            # TTT test time training.
            self.ttt.train()
            ttt_loss = self.ttt(batch.bert_inputs, batch.unseen_inputs, len(domain2slot[batcher[0].domain]), len(domain2unseen[self.args.tgt_dm]), self.vocabs['binary'].inst2idx('O'), batch.token_mask)
            logger.info(f'Test Time Training: [Iter {i}], loss: {ttt_loss}')

            self.ttt.eval()
            batch_bio_pred, batch_slu_pred = self.ttt.evaluate(batch.bert_inputs, len(domain2slot[batcher[0].domain]), len(domain2unseen[self.args.tgt_dm]), batch.token_mask)
            seq_len = batch.token_mask.sum(dim=-1)
            
            # 记录batch内每个pred slot label
            for j, (bio_pred, slu_pred) in enumerate(zip(batch_bio_pred, batch_slu_pred)):
                bio_gold_tags.extend([x.split('-')[0] if x != 'O' else x for x in batcher[j].slu_tags])
                slu_gold_tags.extend([x.split('-')[0] + '-' + desp2slot[x.split('-')[1]] if x != 'O' else x for x in batcher[j].slu_tags])

                sent_bio_pred = self.vocabs['binary'].idx2inst(bio_pred[:seq_len[j]].tolist())
                # sent_slu_pred = [domain2slot[batcher[j].domain][x] for x in slu_pred[:seq_len[j]].tolist()]
                # sent_slu_pred = [x+'-'+y if x != 'O' and y != '[PAD]' else 'O' for x, y in zip(sent_bio_pred, sent_slu_pred)]
                sent_slu_pred = []
                prev = None
                for _bio, _slu in zip(sent_bio_pred, slu_pred[:seq_len[j]].tolist()):
                    slotname = domain2slot[batcher[j].domain][_slu]
                    if slotname == '[PAD]':
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

# def train():
#     # TODO: wandb config
#     args = args_config()
#     if torch.cuda.is_available() and args.cuda >= 0:
#         args.device = torch.device('cuda', args.cuda)
#     else:
#         args.device = torch.device('cpu')
    
#     # TODO: wandb
#     run = wandb.init(project='backbone-cl-domain-sweep', config = {}, entity='switchsyj')
#     wconfig = wandb.config
#     run.name = f"cl_temp:{wconfig['cl_temperature']}"
#     # run.name = f"seed: {wconfig['seed']}, dropout: {wconfig['dropout']}, cl_type:{wconfig['cl_type']}, cl_temp:{wconfig['cl_temperature']}"
#     wconfig['seed'] = 1314
#     args.cl_type = 'cosine'
#     args.dropout = 0.1
#     # args.dropout = wconfig['dropout']
#     # args.cl_type = wconfig['cl_type']
#     args.cl_temperature = wconfig['cl_temperature']
#     set_seeds(wconfig['seed'])
#     print(f"set seed: {wconfig['seed']}")
    
#     # set_seeds(1314)
#     # print(f"set seed: {1314}")
#     trainer = Trainer(args, data_config=None)
#     trainer.train()
#     del trainer


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    
    # TODO: wandb
    # with open('config/sweep_backbonecl.yaml') as f:
    #     wandb_dict = yaml.load(f, Loader=yaml.FullLoader)
    # sweep_id = wandb.sweep(wandb_dict, project='backbone-cl-domain-sweep')
    # wandb.agent(sweep_id, train)
    
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
        # break
    final_res['p'] = np.array(final_res['p'])
    final_res['r'] = np.array(final_res['r'])
    final_res['f'] = np.array(final_res['f'])
    print(f"avg result: p: {final_res['p'].mean()}+-{final_res['p'].std()}, r: {final_res['r'].mean()}+-{final_res['r'].std()}, f: {final_res['f'].mean()}+-{final_res['f'].std()}")

    '''
    logger.info('Final Result: %s' % final_res)
    final_p = sum(final_res['p']) / len(final_res['p'])
    final_r = sum(final_res['r']) / len(final_res['r'])
    final_f = sum(final_res['f']) / len(final_res['f'])
    logger.info('Final P: %.4f, R: %.4f, F: %.4f' % (final_p, final_r, final_f))
    '''


# train cl
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_atp0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_atp0_vocab.ckpt &> training_log/end2end_cl/bert_domain_atp0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt ckpt/end2end_cl/bert_domain_br0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_br0_vocab.ckpt &> training_log/end2end_cl/bert_domain_br0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt ckpt/end2end_cl/bert_domain_gw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_gw0_vocab.ckpt &> training_log/end2end_cl/bert_domain_gw0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt ckpt/end2end_cl/bert_domain_pm0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_pm0_vocab.ckpt &> training_log/end2end_cl/bert_domain_pm0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt ckpt/end2end_cl/bert_domain_rb0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_rb0_vocab.ckpt &> training_log/end2end_cl/bert_domain_rb0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt ckpt/end2end_cl/bert_domain_scw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_scw0_vocab.ckpt &> training_log/end2end_cl/bert_domain_scw0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 1.0 --model_ckpt ckpt/end2end_cl/bert_domain_sse0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_sse0_vocab.ckpt &> training_log/end2end_cl/bert_domain_sse0.log &

# train end2end
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.3 --model_ckpt ckpt/end2end/bert_domain_atp0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_atp0_vocab.ckpt &> training_log/end2end/bert_domain_atp0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.3 --model_ckpt ckpt/end2end/bert_domain_br0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_br0_vocab.ckpt &> training_log/end2end/bert_domain_br0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 3 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.3 --model_ckpt ckpt/end2end/bert_domain_gw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_gw0_vocab.ckpt &> training_log/end2end/bert_domain_gw0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.3 --model_ckpt ckpt/end2end/bert_domain_pm0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_pm0_vocab.ckpt &> training_log/end2end/bert_domain_pm0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.3 --model_ckpt ckpt/end2end/bert_domain_rb0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_rb0_vocab.ckpt &> training_log/end2end/bert_domain_rb0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.3 --model_ckpt ckpt/end2end/bert_domain_scw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_scw0_vocab.ckpt &> training_log/end2end/bert_domain_scw0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.3  --model_ckpt ckpt/end2end/bert_domain_sse0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_sse0_vocab.ckpt &> training_log/end2end/bert_domain_sse0.log &


# test
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.3 --cl --cl_type kl --cl_temperature 2.0 --model_ckpt testlog/test_atp.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_atp0.log &

# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.3 --cl --cl_type kl --cl_temperature 2.0 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_br0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.3 --cl --cl_type kl --cl_temperature 2.0 --model_ckpt testlog/test_gw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_gw0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.3 --cl --cl_type kl --cl_temperature 2.0 --model_ckpt testlog/test_pm.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_pm0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.3 --cl --cl_type kl --cl_temperature 2.0 --model_ckpt testlog/test_rb.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_rb0.log &

# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 3 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.3 --cl --cl_type kl --cl_temperature 2.0 --model_ckpt testlog/test_scw.ckpt --vocab_ckpt ckpt/vocab/test_vocab.ckpt &> testlog/bert_domain_cl_scw0.log &
# nohup python slu_e2e_bert_domain_unseen_train.py --cuda 3 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.3 --cl --cl_type kl --cl_temperature 2.0 --model_ckpt testlog/test_sse.ckpt --vocab_ckpt ckpt/vocab/testsse_vocab.ckpt &> testlog/bert_domain_cl_sse0.log &