import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.e2e_bert_f2tagger import End2endSLUTagger
from modules.testTimeTraining import TTT, TTA, TTDistill
# from weighting.MGDA import MGDA
# from weighting.DWA import DWA
# from weighting.UW import UW
from config.xslu_conf import args_config
from utils.dataset import DataLoader, MultiDomainDataLoader
from utils.xDataset_datautil import load_data, create_vocab, batch_variable, save_to, slot2desp_textbook
from utils.conlleval import evaluate
from logger.logger import logger
from copy import deepcopy
import collections
import json

# TODO: wandb
# import wandb
# import yaml


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        print(self.args)
        
        self.all_dataset = load_data(args.src, args.tgt)
        # TODO: multi dataset transfer.
        
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
        # TODO: add special token [ENT]
        # print(self.vocabs['bert'].bert2id(['[ENT]']))
        # self.vocabs['bert'].bert_tokenizer.add_tokens(['[ENT]'], special_tokens=True)   
        # print(self.vocabs['bert'].bert2id(['[ENT]']))
        # save_to(args.vocab_ckpt, self.vocabs)

        print([x for x in self.vocabs["binary"]])
        self.domain2slot = dict()
        # with open(f"data/{args.src}/labels.json", 'r') as f:
        with open(f"data/merge_dataset_leona/{args.src}/labels.json", 'r') as f:
            _temp = json.load(f)
            for k, v in _temp.items():
                self.domain2slot[k] = ['[PAD]'] + [x for x in v if x != 'O']
            del _temp
        print("======domain2slot for source domain: ======\n", self.domain2slot)
        print(f"****** average slots number: {np.mean(np.array([len(x) for _, x in self.domain2slot.items()]))} ******")
        with open(f"data/dataset/{args.tgt}/labels.json", 'r') as f:
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
                # self.slot2desp[slot] = slot
        print(f"******Loading {len(overall_slots)} slots for training and testing ******")
        
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
            alpha=args.alpha,
            beta=args.beta,
            ).to(args.device)
        
        # TODO: add special token [ENT]
        # self.slu_model.bert.bert.resize_token_embeddings(len(self.vocabs['bert'].bert_tokenizer))
        
        print(self.slu_model)
        total_params = sum([p.numel() for gen in [self.slu_model.parameters()] for p in gen if p.requires_grad])
        print("Training %dM trainable parameters..." % (total_params/1e6))
        
        # self.weighted_loss = UW(task_name=['main_loss', 'orth_loss', 'cl_loss'], device=self.args.device)
        
        no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_parameters = [
        #     {'params': [p for n, p in self.slu_model.bert_named_params()
        #                 if not any(nd in n for nd in no_decay) and p.requires_grad],
        #      'weight_decay': 1e-2, 'lr': self.args.bert_lr},
        #     {'params': [p for n, p in self.slu_model.bert_named_params()
        #                 if any(nd in n for nd in no_decay) and p.requires_grad],
        #      'weight_decay': 0.0, 'lr': self.args.bert_lr},
            
        #     {'params': [p for n, p in self.slu_model.base_named_params() if not any(nd in n for nd in no_decay)],
        #      'weight_decay': 1e-4, 'lr': self.args.lr},
        #     {'params': [p for n, p in self.slu_model.base_named_params() if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0, 'lr': self.args.lr},
            
        # ]
        
        # TODO: VAE param
        # labelspace_param = ['hidden2mean', 'hidden2logv', 'latent2hidden']
        # TODO: Adapter param
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

    # def get_mask_p(self, ep):
    #     # linear warmup with k=0.3
    #     # return (sub_p, mask_p)
    #     if ep > self.ep:
    #         # return min(0.1 * (ep - 4), 0.5)
    #         return min(self.step * (ep - self.ep), self.ceil)
    #     else:
    #         return 0.

    def train_epoch(self, ep=0):
        self.slu_model.train()
        t1 = time.time()
        loss_tr = {"bio_loss": 0., "slu_loss": 0., "cl_loss": 0., "orth_loss": 0., "utter": 0., "label": 0., "vae_kl": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            # mask_p = self.get_mask_p(ep-1)
            batch = batch_variable(batch_train_data, self.vocabs, self.slot2desp, self.domain2slot, mode='train')
            batch.to_device(self.args.device)
            
            O_tag_idx = self.vocabs['binary'].inst2idx('O')
            loss, dict = self.slu_model(batch.bert_inputs,
                                        len(self.domain2slot[batch_train_data[0].domain]),
                                        batch.bio_label, batch.slu_label, O_tag_idx, 
                                        mask=batch.token_mask,
                                        train='utter',
                                        step=ep*len(self.train_loader)+i, x0=(len(self.train_loader)*self.args.epoch)//10,
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
                            len(self.domain2slot[batch_train_data[0].domain]),
                            batch.bio_label, batch.slu_label, O_tag_idx, 
                            mask=batch.token_mask,
                            train='label',
                            step=ep*len(self.train_loader)+i, x0=(len(self.train_loader)*self.args.epoch)//10,
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
        self.ep = 2
        self.step = 0.02
        self.ceil = 0.1
        
        patient = 0
        best_dev_metric, best_test_metric = dict(), dict()
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.train_epoch(ep)

            dev_metric = self.evaluate(self.dev_loader)
            test_metric = self.evaluate(self.test_loader)
            if dev_metric['slu_f'] > best_dev_metric.get('slu_f', 0):
                best_dev_metric = dev_metric
                # self.save_states(self.args.model_ckpt, best_dev_metric)
                inf_st_time = time.time()
                best_test_metric = self.evaluate(self.test_loader)
                print(f"======inference time: {time.time()-inf_st_time}======")
                
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

    def evaluate(self, test_loader, output_emb=False):
        """
        Dev set evaluation
        """
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        # desp2slot = {v: k for k, v in self.slot2desp.items()}
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
                    # sent_slu_pred = [domain2slot[batcher[j].domain][x] for x in slu_pred[:seq_len[j]].tolist()]
                    # sent_slu_pred = [x+'-'+y if x != 'O' and y != '[PAD]' else 'O' for x, y in zip(sent_bio_pred, sent_slu_pred)]
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
        break
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

# nohup python f2xDataset.py --cuda 6 -lr 1e-4 --src atis --tgt snips --batch_size 8 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/atis-snips.ckpt --vocab_ckpt testlog/atis-snips_vocab.ckpt &> testlog/atis-snips05drop01type01.log &

# nohup python f2xDataset.py --cuda 6 -lr 1e-3 --src snips --tgt atis --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/snips-atis_merge_new.log &
# nohup python f2xDataset.py --cuda 3 -lr 1e-3 --src snips --tgt multiwoz --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/snips-woz.ckpt --vocab_ckpt testlog/snips-woz_vocab.ckpt &> testlog/snips-woz.log &
# nohup python f2xDataset.py --cuda 3 -lr 1e-3 --src snips --tgt sgd --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/snips-sgd.ckpt --vocab_ckpt testlog/snips-sgd_vocab.ckpt &> testlog/snips-sgd_merge.log &

# nohup python f2xDataset.py --cuda 7 -lr 1e-3 --src atis --tgt snips --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/atis-snips.ckpt --vocab_ckpt testlog/atis-snips_vocab.ckpt &> testlog/atis-snips_merge_new.log &
# nohup python f2xDataset.py --cuda 4 -lr 1e-3 --src atis --tgt multiwoz --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/atis-woz.ckpt --vocab_ckpt testlog/atis-woz_vocab.ckpt &> testlog/atis-woz.log &
# nohup python f2xDataset.py --cuda 4 -lr 1e-3 --src atis --tgt sgd --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/atis-sgd.ckpt --vocab_ckpt testlog/atis-sgd_vocab.ckpt &> testlog/atis-sgd_merge.log &

# nohup python f2xDataset.py --cuda 6 -lr 1e-3 --src multiwoz --tgt snips --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/multiwoz-snips.ckpt --vocab_ckpt testlog/multiwoz-snips.ckpt &> testlog/multiwoz-snips.log &
# nohup python f2xDataset.py --cuda 6 -lr 1e-3 --src multiwoz --tgt atis --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/multiwoz-atis.ckpt --vocab_ckpt testlog/multiwoz-atis_vocab.ckpt &> testlog/multiwoz-atis.log &
# nohup python f2xDataset.py --cuda 6 -lr 1e-3 --src multiwoz --tgt sgd --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/multiwoz-sgd.ckpt --vocab_ckpt testlog/multiwoz-sgd_vocab.ckpt &> testlog/multiwoz-sgd.log &

# nohup python f2xDataset.py --cuda 3 -lr 1e-3 --src sgd --tgt snips --batch_size 128 --val_batch_size 128 --test_batch_size 128 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/sgd-snips.ckpt --vocab_ckpt testlog/sgd-snips_vocab.ckpt &> testlog/sgd-snips_merge.log &
# nohup python f2xDataset.py --cuda 4 -lr 1e-3 --src sgd --tgt atis --batch_size 128 --val_batch_size 128 --test_batch_size 128 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/sgd-atis.ckpt --vocab_ckpt testlog/sgd-atis_vocab.ckpt &> testlog/sgd-atis_merge.log &
# nohup python f2xDataset.py --cuda 6 -lr 1e-3 --src sgd --tgt multiwoz --batch_size 32 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/sgd-multiwoz.ckpt --vocab_ckpt testlog/sgd-multiwoz_vocab.ckpt &> testlog/sgd-multiwoz.log &

# nohup python f2xDataset.py --cuda 0 -lr 1e-3 --src snips --tgt sgd --val_batch_size 128 --test_batch_size 128 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/snips-sgd.ckpt --vocab_ckpt testlog/snips-sgd_vocab.ckpt &> testlog/snips-sgd_merge.log &
# nohup python f2xDataset.py --cuda 1 -lr 1e-3 --src snips --tgt atis --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/snips-atis_merge.log &
# nohup python f2xDataset.py --cuda 2 -lr 1e-4 --src atis --tgt sgd --batch_size 8 --val_batch_size 128 --test_batch_size 128 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/atis-sgd.ckpt --vocab_ckpt testlog/atis-sgd_vocab.ckpt &> testlog/atis-sgd_merge.log &
# nohup python f2xDataset.py --cuda 3 -lr 1e-4 --src atis --tgt snips --batch_size 8 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/atis-snips.ckpt --vocab_ckpt testlog/atis-snips_vocab.ckpt &> testlog/atis-snips_merge.log &
# nohup python f2xDataset.py --cuda 4 -lr 1e-3 --src sgd --tgt snips --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/sgd-snips.ckpt --vocab_ckpt testlog/sgd-snips_vocab.ckpt &> testlog/sgd-snips_merge.log &
# nohup python f2xDataset.py --cuda 5 -lr 1e-3 --src sgd --tgt atis --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/sgd-atis.ckpt --vocab_ckpt testlog/sgd-atis_vocab.ckpt &> testlog/sgd-atis_merge.log &

# nohup python f2xDataset.py --cuda 1 -lr 1e-3 --src snips --tgt atis --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/snips-atis_merge_des.log &
# nohup python f2xDataset.py --cuda 3 -lr 1e-4 --src atis --tgt snips --batch_size 8 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/atis-snips.ckpt --vocab_ckpt testlog/atis-snips_vocab.ckpt &> testlog/atis-snips_merge_des.log &
# nohup python f2xDataset.py --cuda 0 -lr 1e-4 --src atis --tgt sgd --batch_size 8 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/atis-sgd.ckpt --vocab_ckpt testlog/atis-sgd_vocab.ckpt &> testlog/atis-sgd_merge_des.log &

# tar -zcvf arxiv.tar.gz --exclude='./arxiv/1007' --exclude='./arxiv/2303' --exclude='./arxiv/2304' --exclude='./arxiv/random_selected_sample' --exclude='./arxiv/processed_arxiv' --exclude='./arxiv/processed_arxiv2' --exclude='./arxiv/processed_arxiv3' --exclude='./arxiv/logs' --exclude='./arxiv/paimon-latest.tar.gz' --exclude='./arxiv/arxiv_src.jsonl' --exclude='./arxiv/arxiv_src-2.jsonl' --exclude='./arxiv/arxiv_src-2-filter.jsonl' --exclude='./arxiv/arxiv_src-2-filter-dedup.jsonl' ./arxiv
# tar -zcvf arxiv.tar.gz --exclude=./arxiv/1007 --exclude=./arxiv/2303 --exclude=./arxiv/2304 --exclude=./arxiv/random_selected_sample --exclude=./arxiv/processed_arxiv --exclude=./arxiv/processed_arxiv2 --exclude=./arxiv/processed_arxiv3 --exclude=./arxiv/logs --exclude=./arxiv/paimon-latest.tar.gz --exclude=./arxiv/arxiv_src.jsonl --exclude=./arxiv/arxiv_src-2.jsonl --exclude=./arxiv/arxiv_src-2-filter.jsonl --exclude=./arxiv/arxiv_src-2-filter-dedup.jsonl ./arxiv

# nohup python f2xDataset.py --cuda 1 -lr 1e-3 --src snips --tgt atis --alpha 1.5 --beta 1.0 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/snips-atis_merge_test.log &
# nohup python f2xDataset.py --cuda 3 -lr 1e-3 --src snips --tgt sgd --alpha 1.5 --beta 1.5 --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/snips-sgd_merge_newdev.log &
# nohup python f2xDataset.py --cuda 1 -lr 1e-4 --src atis --tgt snips --batch_size 8 --alpha 2.0 --beta 2.0 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/atis-snips_merge_test.log &
# nohup python f2xDataset.py --cuda 3 -lr 1e-4 --src atis --tgt sgd --batch_size 8 --alpha 2.0 --beta 1.5 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/atis-sgd_merge_newdev.log &
# nohup python f2xDataset.py --cuda 2 -lr 1e-3 --src sgd --tgt snips --alpha 2.0 --beta 1.5 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/sgd-snips_merge_test.log &
# nohup python f2xDataset.py --cuda 3 -lr 1e-3 --src sgd --tgt atis --alpha 2.0 --beta 1.5 --epoch 30 --dropout 0.1 --cl --cl_type euclidean --cl_temperature 0.5 --model_ckpt testlog/snips-atis.ckpt --vocab_ckpt testlog/snips-atis_vocab.ckpt &> testlog/sgd-atis_merge_test.log &