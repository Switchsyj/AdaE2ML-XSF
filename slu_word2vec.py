import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.word2vec import End2endSLUTagger
# from weighting.MGDA import MGDA
# from weighting.DWA import DWA
# from weighting.UW import UW
from config.slu_conf import args_config
from utils.dataset import DataLoader, MultiDomainDataLoader
from utils.word2vec_datautil import load_data, create_vocab, batch_variable, save_to, domain2slot, slot2desp, domain2unseen
from utils.conlleval import evaluate
from logger.logger import logger

# TODO: wandb
# import wandb
# import yaml


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

        self.vocabs = create_vocab(self.all_dataset, args.bert_path, '/root/autodl-tmp/zero-shot-slu/data/snips/cache/glove.840B.300d.txt')
        # save_to(args.vocab_ckpt, self.vocabs)

        print([x for x in self.vocabs["binary"]])
        
        self.slu_model = End2endSLUTagger(
            self.vocabs,
            bert_embed_dim=self.args.emb_dim,
            num_bound=len(self.vocabs['binary']),
            num_bert_layer=4,
            dropout=args.dropout,
            use_cl=args.cl,
            cl_type=args.cl_type,
            bert_model_path='bert_model/bert-base-uncased',
            cl_temperature=args.cl_temperature,
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
            
            {'params': [p for n, p in self.slu_model.base_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-4, 'lr': self.args.lr},
            {'params': [p for n, p in self.slu_model.base_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.lr},
            
        ]
        # optimizer_parameters = [
        #     {'params': [p for n, p in self.slu_model.named_parameters()
        #                 if not any(nd in n for nd in no_decay) and p.requires_grad],
        #     'weight_decay': 1e-2, 'lr': self.args.lr},
        #     {'params': [p for n, p in self.slu_model.named_parameters()
        #                 if any(nd in n for nd in no_decay) and p.requires_grad],
        #      'weight_decay': 0.0, 'lr': self.args.lr}
        # ]
        
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.args.lr, eps=1e-8)
        max_step = len(self.train_loader) * self.args.epoch
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=max_step // 15, t_total=max_step)

    def train_epoch(self, ep=0):
        self.slu_model.train()
        t1 = time.time()
        loss_tr = {"bio_loss": 0., "slu_loss": 0., "cl_loss": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            batch = batch_variable(batch_train_data, self.vocabs, self.args.tgt_dm, mode='train')
            batch.to_device(self.args.device)
            
            O_tag_idx = self.vocabs['binary'].inst2idx('O')
            loss, dict = self.slu_model(batch.bert_inputs,
                                        batch.bert_label_inputs,
                                        batch.glove_label_inputs,
                                        len(domain2slot[batch_train_data[0].domain]),
                                        batch.bio_label, batch.slu_label, O_tag_idx, 
                                        mask=batch.token_mask
                                        )
            loss_tr["bio_loss"] = dict["bio_loss"]
            loss_tr["slu_loss"] = dict["slu_loss"]
            loss_tr["cl_loss"] = dict["cl_loss"]
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.slu_model.zero_grad()
            
            # TODO: wandb
            # wandb.log({
            #     "slu_loss": loss.item(),
            #     "lr": self.scheduler.get_last_lr()[-1]
            # })

            logger.info('[Epoch %d] [lr: %.4f] Iter%d time cost: %.4fs, loss: %.4f, binary_loss: %.4f, slu_loss: %.4f, cl_loss: %.4f' % \
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
        # TODO: wandb
        # run = wandb.init(project='backbone-cl-sweep', config = {}, entity='switchsyj')
        # wconfig = wandb.config
        # run.name = f"ep: {wconfig['ep']}, step: {wconfig['step']}, ceil: {wconfig['ceil']}"
        # self.ep = wconfig['ep']
        # self.step = wconfig['step']
        # self.ceil = wconfig['ceil']
        
        patient = 0
        best_dev_metric, best_test_metric = dict(), dict()
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.train_epoch(ep)

            dev_metric = self.evaluate(self.dev_loader)
            # test_metric = self.evaluate(self.test_loader)
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
                
                batch_bio_pred, batch_slu_pred = self.slu_model.evaluate(batch.bert_inputs,
                                                                         batch.bert_label_inputs,
                                                                         batch.glove_label_inputs,
                                                                         len(domain2slot[batcher[0].domain]), 
                                                                         batch.token_mask)
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
#     set_seeds(1314)
#     print(f"set seed: {1314}")
#     trainer = Trainer(args, data_config=None)
#     trainer.train()
#     del trainer


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    
    # TODO: wandb
    # with open('config/sweep.yaml') as f:
    #     wandb_dict = yaml.load(f, Loader=yaml.FullLoader)
    # sweep_id = wandb.sweep(wandb_dict, project='backbone-cl-sweep')
    # wandb.agent(sweep_id, train)
    
    args = args_config()
    
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')
    
    random_seed = [1314, 2019, 2020, 2021, 2022, 2023]
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


# train cl
# nohup python slu_word2vec.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --emb_dim 768 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_cl_atp0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_atp0_vocab.ckpt &> training_log/word2vec/bert_domain_atp0.log &
# nohup python slu_word2vec.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --emb_dim 768 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_cl_br0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_br0_vocab.ckpt &> training_log/word2vec/bert_domain_br0.log &
# nohup python slu_word2vec.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --emb_dim 768 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_cl_gw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_gw0_vocab.ckpt &> training_log/word2vec/bert_domain_gw0.log &
# nohup python slu_word2vec.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --emb_dim 768 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_cl_pm0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_pm0_vocab.ckpt &> training_log/word2vec/bert_domain_pm0.log &
# nohup python slu_word2vec.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --emb_dim 768 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_cl_rb0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_rb0_vocab.ckpt &> training_log/word2vec/bert_domain_rb0.log &
# nohup python slu_word2vec.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --emb_dim 768 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_cl_scw0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_scw0_vocab.ckpt &> training_log/word2vec/bert_domain_scw0.log &
# nohup python slu_word2vec.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --emb_dim 768 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_cl_sse0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_cl_sse0_vocab.ckpt &> training_log/word2vec/bert_domain_sse0.log &