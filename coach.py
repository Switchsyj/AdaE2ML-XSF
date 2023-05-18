import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from model.coach_tagger import BinarySLUTagger, SentRepreGenerator, SlotNamePredictor
from config.slu_conf import args_config
from utils.coach_datautil import load_data, create_vocab, batch_variable, save_to
from data.snips.generate_slu_emb import domain2slot
from utils.conlleval import evaluate
from logger.logger import logger
import pickle as pkl


class Dataset(Dataset):
    def __init__(self, tokens, templates, domains, bio_label, slu_label):
        self.tokens = tokens
        self.templates = templates
        self.domains = domains
        self.bio_label = bio_label
        self.slu_label = slu_label

    def __getitem__(self, index):
        return self.tokens[index], self.templates[index], self.domains[index], self.bio_label[index], self.slu_label[index]

    def __len__(self):
        return len(self.tokens)

def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False
    
def move_to_device(obj, device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """
    if not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj
    
def my_collate_fn(batch_data, Vocab):
    tokens_list, templates_list, domains_list, bio_label_list, slu_label_list = zip(*batch_data)
    bsz = len(tokens_list)
    token_max_len = max(len(token) for token in tokens_list)
    token_seq_len = [len(token) for token in tokens_list]
    template_max_len = max(len(template) for templates in templates_list for template in templates)
    template_seq_len = [len(template) for templates in templates_list for template in templates]
    
    token_vocab = Vocab['token']
    slu_vocab = Vocab['slot']
    binary_vocab = Vocab['binary']
    domain_vocab = Vocab['domain']
    
    # filled with padding idx
    token_ids = torch.zeros((bsz, token_max_len), dtype=torch.long)
    template_ids = torch.zeros((bsz*3, template_max_len), dtype=torch.long)
    domain_ids = torch.zeros((bsz, 1), dtype=torch.long)
    
    bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    token_mask = torch.zeros((bsz, token_max_len), dtype=torch.bool)
    template_mask = torch.zeros((bsz*3, template_max_len), dtype=torch.float)
    
    for i in range(bsz):
        tokens, templates, bio_list, slu_list, domain = tokens_list[i], templates_list[i], bio_label_list[i], slu_label_list[i], domains_list[i]
        token_ids[i, :token_seq_len[i]] = torch.tensor(token_vocab.inst2idx(tokens), dtype=torch.long)
        token_mask[i, :token_seq_len[i]] = torch.tensor([1. for _ in range(token_seq_len[i])])
        
        template_ids[3*i, :template_seq_len[3*i]] = torch.tensor(token_vocab.inst2idx(templates[0]), dtype=torch.long)
        template_ids[3*i+1, :template_seq_len[3*i+1]] = torch.tensor(token_vocab.inst2idx(templates[1]), dtype=torch.long)
        template_ids[3*i+2, :template_seq_len[3*i+2]] = torch.tensor(token_vocab.inst2idx(templates[2]), dtype=torch.long)
        template_mask[3*i, :template_seq_len[3*i]] = torch.tensor([1. for _ in range(template_seq_len[3*i])])
        template_mask[3*i+1, :template_seq_len[3*i+1]] = torch.tensor([1. for _ in range(template_seq_len[3*i+1])])
        template_mask[3*i+2, :template_seq_len[3*i+2]] = torch.tensor([1. for _ in range(template_seq_len[3*i+2])])

        
        domain_ids[i] = domain_vocab.inst2idx(domain)
        bio_label[i, :token_seq_len[i]] = torch.tensor(binary_vocab.inst2idx(bio_list), dtype=torch.long)
        slu_label[i, :token_seq_len[i]] = torch.tensor(slu_vocab.inst2idx(slu_list), dtype=torch.long)
    
    return token_ids, template_ids, domain_ids, bio_label, slu_label, token_mask, template_mask


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        print(self.args)
        
        self.all_dataset = load_data()
        self.tgt_dm = args.tgt_dm
        self.n_sample = args.n_sample
        
        train_set = {"tokens": [], "templates": [], "domains": [], "bio_label": [], "slu_label": []}
        val_set = {"tokens": [], "templates": [], "domains": [], "bio_label": [], "slu_label": []}
        test_set = {"tokens": [], "templates": [], "domains": [], "bio_label": [], "slu_label": []}
        
        for dm, insts in self.all_dataset.items():
            if dm != self.tgt_dm:
                for inst in insts:
                    train_set["tokens"].append(inst.tokens)
                    train_set["templates"].append(inst.template_list)
                    train_set["domains"].append(inst.domain)
                    train_set["bio_label"].append(inst.binary_tags)
                    train_set["slu_label"].append(inst.slu_tags)
            else:
                for i, inst in enumerate(insts):
                    if i < self.n_sample:
                        train_set["tokens"].append(inst.tokens)
                        train_set["templates"].append(inst.template_list)
                        train_set["domains"].append(inst.domain)
                        train_set["bio_label"].append(inst.binary_tags)
                        train_set["slu_label"].append(inst.slu_tags)
                    elif i >= self.n_sample and i < 500:
                        val_set["tokens"].append(inst.tokens)
                        val_set["templates"].append(inst.template_list)
                        val_set["domains"].append(inst.domain)
                        val_set["bio_label"].append(inst.binary_tags)
                        val_set["slu_label"].append(inst.slu_tags)
                    else:
                        test_set["tokens"].append(inst.tokens)
                        test_set["templates"].append(inst.template_list)
                        test_set["domains"].append(inst.domain)
                        test_set["bio_label"].append(inst.binary_tags)
                        test_set["slu_label"].append(inst.slu_tags)
 
        print('train data size:', len(train_set['tokens']))
        print('validate data size:', len(val_set['tokens']))
        print('test data size:', len(test_set['tokens']))
        
        # TODO: change vocabulary
        with open("data/snips/cache/revised/vocab_ckpt.pkl", 'rb') as f:
            self.vocabs = pkl.load(f)
            print(self.vocabs["binary"]._inst2idx)
            print(self.vocabs["slot"]._inst2idx)
            print(self.vocabs["domain"]._inst2idx)
        
        self.train_set = Dataset(tokens=train_set['tokens'], templates=train_set['templates'], domains=train_set['domains'], bio_label=train_set['bio_label'], slu_label=train_set['slu_label'])
        self.val_set = Dataset(tokens=val_set['tokens'], templates=val_set['templates'], domains=val_set['domains'], bio_label=val_set['bio_label'], slu_label=val_set['slu_label'])
        self.test_set = Dataset(tokens=test_set['tokens'], templates=test_set['templates'], domains=test_set['domains'], bio_label=test_set['bio_label'], slu_label=test_set['slu_label'])
        
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: my_collate_fn(x, self.vocabs))
        self.dev_loader = DataLoader(dataset=self.val_set, batch_size=self.args.batch_size, shuffle=False, collate_fn=lambda x: my_collate_fn(x, self.vocabs))
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.args.batch_size, shuffle=False, collate_fn=lambda x: my_collate_fn(x, self.vocabs))
        
        self.binary_slu_tagger = BinarySLUTagger(emb_dim=self.args.emb_dim, freeze_emb=self.args.freeze_emb, emb_file=self.args.emb_file, num_layers=self.args.num_rnn_layer, hidden_size=self.args.hidden_size, dropout=self.args.dropout, bidirectional=self.args.bidirectional, vocabs=self.vocabs).to(self.args.device)
        self.slotname_predictor = SlotNamePredictor(hidden_size=self.args.hidden_size, bidirectional=self.args.bidirectional, trs_hidden_size=self.args.hidden_size*2, trs_layers=1, slot_emb_file=self.args.slot_emb_file, vocabs=self.vocabs).to(self.args.device)
        self.sent_repre_generator = SentRepreGenerator(emb_dim=self.args.emb_dim, freeze_emb=self.args.freeze_emb, emb_file=self.args.emb_file, hidden_size=self.args.hidden_size, num_layers=self.args.num_rnn_layer, dropout=self.args.dropout, bidirectional=self.args.bidirectional, vocabs=self.vocabs).to(self.args.device)
        
        model_parameters = [
            {"params": self.binary_slu_tagger.parameters()},
            {"params": self.slotname_predictor.parameters()},
            {"params": self.sent_repre_generator.parameters()}
        ]
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.args.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_mse = nn.MSELoss()
        
    def save_states(self, save_path, best_metric=None):
        self.binary_slu_tagger.zero_grad()
        self.slotname_predictor.zero_grad()
        self.sent_repre_generator.zero_grad()
        check_point = {'binary_model_state': self.binary_slu_tagger.state_dict(),
                       'slot_model_state': self.slotname_predictor.state_dict(),
                       'tr_model_state': self.sent_repre_generator.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'best_prf': best_metric,
                       'args': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')
    
    def train_epoch(self, ep):
        self.binary_slu_tagger.train()
        self.slotname_predictor.train()
        self.sent_repre_generator.train()
        
        loss_tr = {"bio_loss": 0., "sent_slu_loss": 0., "tem0_loss": 0., "tem1_loss": 0.}
        for i, batch_train_data in enumerate(self.train_loader):
            batch_train_data = move_to_device(batch_train_data, self.args.device)
            tokens, templates, domains, bio_label, slu_label, token_mask, template_mask = batch_train_data
            t1 = time.time()
        
            bin_preds, lstm_hiddens = self.binary_slu_tagger(tokens)

            ## optimize binary_slu_tagger
            loss_bin = self.binary_slu_tagger.crf_loss(bin_preds, bio_label)
            self.optimizer.zero_grad()
            loss_bin.backward()
            self.optimizer.step()
            loss_tr['bio_loss'] = loss_bin.item()

            _, lstm_hiddens = self.binary_slu_tagger(tokens)
            ## optimize slotname_predictor
            pred_slotname_list, gold_slotname_list = self.slotname_predictor(domains, lstm_hiddens, binary_golds=bio_label, final_golds=slu_label)
            loss_slotname = 0.
            for pred_slotname_each_sample, gold_slotname_each_sample in zip(pred_slotname_list, gold_slotname_list):
                assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                loss_slotname += self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.to(tokens.device)) / len(pred_slotname_list)
                
            self.optimizer.zero_grad()
            loss_slotname.backward()
            self.optimizer.step()
            loss_tr['sent_slu_loss'] = loss_slotname.item()
            
            _, lstm_hiddens = self.binary_slu_tagger(tokens)
            templates_repre, input_repre = self.sent_repre_generator(templates, template_mask.sum(1), lstm_hiddens, token_mask.sum(1))

            input_repre = input_repre.detach()
            template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
            template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
            template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
            input_repre.requires_grad = True

            # self.optimizer.zero_grad()
            # template0_loss.backward(retain_graph=True)
            # template1_loss.backward(retain_graph=True)
            # template2_loss.backward(retain_graph=True)
            # self.optimizer.step()
            template_loss = template0_loss + template1_loss + template2_loss
            self.optimizer.zero_grad()
            template_loss.backward()
            self.optimizer.step()
            
            loss_tr['tem0_loss'] = template0_loss.item()
            loss_tr['tem1_loss'] = template1_loss.item()

            if ep > 4:
                templates_repre = templates_repre.detach()
                input_loss0 = self.loss_fn_mse(input_repre, templates_repre[:, 0, :])
                input_loss1 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 1, :])
                input_loss2 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 2, :])
                templates_repre.requires_grad = True

                # self.optimizer.zero_grad()
                # input_loss0.backward(retain_graph=True)
                # input_loss1.backward(retain_graph=True)
                # input_loss2.backward(retain_graph=True)
                # self.optimizer.step()
                input_loss = input_loss0 + input_loss1 + input_loss2
                self.optimizer.zero_grad()
                input_loss.backward()
                self.optimizer.step()
            
            logger.info('[Epoch %d] Iter%d time cost: %.4fs, bio loss: %.4f, slu_loss: %.4f, tem0_loss: %.4f, tem1_loss: %.4f' % \
                (ep, i, time.time() - t1, loss_bin.item(), loss_slotname.item(), template0_loss.item(), template1_loss.item()))
        
        return loss_tr

    def train(self):
        patient = 0
        best_dev_metric, best_test_metric = dict(), dict()
        for ep in range(1, 1+self.args.epoch):
            train_loss = self.train_epoch(ep)

            dev_metric = self.evaluate(self.dev_loader)
            if dev_metric['slu_f'] > best_dev_metric.get('slu_f', 0):
                best_dev_metric = dev_metric
                self.save_states(self.args.model_ckpt, best_dev_metric)
                best_test_metric = self.evaluate(self.test_loader)
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

    def combine_binary_and_slotname_preds(self, dm_id_batch, binary_preds_batch, slotname_preds_batch):
        """
        Input:
            dm_id_batch: (bsz)
            binary_preds: (bsz, seq_len)
            slotname_preds: (bsz, num_slotname, slot_num)
        Output:
            final_preds: (bsz, seq_len)
        """
        final_preds = []
        for i in range(len(dm_id_batch)):
            dm_id = dm_id_batch[i].item()
            binary_preds = binary_preds_batch[i]
            slotname_preds = slotname_preds_batch[i]
            slot_list_based_dm = domain2slot[self.vocabs['domain'].idx2inst(dm_id)]
            
            i = -1
            final_preds_each = []
            for bin_pred in binary_preds:
                # values of bin_pred are 0 (O), or 1(B) or 2(I)
                if bin_pred.item() == self.vocabs['binary'].inst2idx('O'):
                    final_preds_each.append('O')
                elif bin_pred.item() == self.vocabs['binary'].inst2idx('B'):
                    i += 1
                    pred_slot_id = torch.argmax(slotname_preds[i])
                    slotname = "B-" + slot_list_based_dm[pred_slot_id]
                    final_preds_each.append(slotname)
                elif bin_pred.item() == self.vocabs['binary'].inst2idx('I'):
                    if i == -1:
                        final_preds_each.append('O')
                    else:
                        pred_slot_id = torch.argmax(slotname_preds[i])
                        slotname = "I-" + slot_list_based_dm[pred_slot_id]
                        if slotname not in self.vocabs['slot']._inst2idx:
                            final_preds_each.append('O')
                        else:
                            final_preds_each.append(slotname)
                else:
                    final_preds_each.append('O')
            assert len(final_preds_each) == len(binary_preds)
            final_preds.append(final_preds_each)

        return final_preds

    def evaluate(self, test_loader):
        self.binary_slu_tagger.eval()
        self.slotname_predictor.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        with torch.no_grad():
            for _, batch_test_data in enumerate(test_loader):
                batch_test_data = move_to_device(batch_test_data, self.args.device)
                tokens, _, domains, bio_label, slu_label, token_mask, _ = batch_test_data

                bin_preds_batch, lstm_hiddens = self.binary_slu_tagger(tokens)
                bin_preds_batch = self.binary_slu_tagger.crf_decode(bin_preds_batch, token_mask.sum(1))

                slotname_preds_batch = self.slotname_predictor(domains, lstm_hiddens, binary_preditions=bin_preds_batch, binary_golds=None, final_golds=None)
                
                final_preds_batch = self.combine_binary_and_slotname_preds(domains, bin_preds_batch, slotname_preds_batch)
                
                seq_len = token_mask.sum(1)
                # 记录batch内每个pred slot label
                for j, (bio_pred, slu_pred) in enumerate(zip(bin_preds_batch, final_preds_batch)):
                    assert len(bio_pred) == len(slu_pred)
                    bio_gold_tags.extend(self.vocabs['binary'].idx2inst(bio_label[j].tolist()[:seq_len[j]]))
                    slu_gold_tags.extend(self.vocabs['slot'].idx2inst(slu_label[j].tolist()[:seq_len[j]]))
                    
                    bio_pred_tags.extend([self.vocabs['binary'].idx2inst(x) for x in bio_pred.tolist()])
                    slu_pred_tags.extend(slu_pred)
                    
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


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    random_seeds = [1314, 2019, 2020, 2021, 2022, 2023]
    
    final_res = {'p': [], 'r': [], 'f': []}
    for seed in random_seeds:
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
    
# nohup python coach.py --cuda 0 -lr 5e-4 --n_sample 0 --tgt_dm AddToPlaylist --freeze_emb --tr --epoch 300 --emb_dim 400 --dropout 0.3 --hidden_size 200 --num_rnn_layer 2 --emb_file data/snips/cache/revised/slu+slot_word_char.npy --slot_emb_file data/snips/cache/revised/slot_word_char.dict --model_ckpt ckpt/coach/coach_atp0.ckpt --vocab_ckpt ckpt/coach/coach_atp0_vocab.ckpt &> training_log/coach/baseline_atp0.log & 
# nohup python coach.py --cuda 1 -lr 5e-4 --n_sample 0 --tgt_dm BookRestaurant --freeze_emb --tr --epoch 300 --emb_dim 400 --dropout 0.3 --hidden_size 200 --num_rnn_layer 2 --emb_file data/snips/cache/revised/slu+slot_word_char.npy --slot_emb_file data/snips/cache/revised/slot_word_char.dict --model_ckpt ckpt/coach/coach_br0.ckpt --vocab_ckpt ckpt/coach/coach_br0_vocab.ckpt &> training_log/coach/baseline_br0.log & 
# nohup python coach.py --cuda 2 -lr 5e-4 --n_sample 0 --tgt_dm GetWeather --freeze_emb --tr --epoch 300 --emb_dim 400 --dropout 0.3 --hidden_size 200 --num_rnn_layer 2 --emb_file data/snips/cache/revised/slu+slot_word_char.npy --slot_emb_file data/snips/cache/revised/slot_word_char.dict --model_ckpt ckpt/coach/coach_gw0.ckpt --vocab_ckpt ckpt/coach/coach_gw0_vocab.ckpt &> training_log/coach/baseline_gw0.log & 
# nohup python coach.py --cuda 0 -lr 5e-4 --n_sample 0 --tgt_dm PlayMusic --freeze_emb --tr --epoch 300 --emb_dim 400 --dropout 0.3 --hidden_size 200 --num_rnn_layer 2 --emb_file data/snips/cache/revised/slu+slot_word_char.npy --slot_emb_file data/snips/cache/revised/slot_word_char.dict --model_ckpt ckpt/coach/coach_pm0.ckpt --vocab_ckpt ckpt/coach/coach_pm0_vocab.ckpt &> training_log/coach/baseline_pm0.log & 
# nohup python coach.py --cuda 1 -lr 5e-4 --n_sample 0 --tgt_dm RateBook --freeze_emb --tr --epoch 300 --emb_dim 400 --dropout 0.3 --hidden_size 200 --num_rnn_layer 2 --emb_file data/snips/cache/revised/slu+slot_word_char.npy --slot_emb_file data/snips/cache/revised/slot_word_char.dict --model_ckpt ckpt/coach/coach_rb0.ckpt --vocab_ckpt ckpt/coach/coach_rb0_vocab.ckpt &> training_log/coach/baseline_rb0.log & 
# nohup python coach.py --cuda 2 -lr 5e-4 --n_sample 0 --tgt_dm SearchCreativeWork --freeze_emb --tr --epoch 300 --emb_dim 400 --dropout 0.3 --hidden_size 200 --num_rnn_layer 2 --emb_file data/snips/cache/revised/slu+slot_word_char.npy --slot_emb_file data/snips/cache/revised/slot_word_char.dict --model_ckpt ckpt/coach/coach_scw0.ckpt --vocab_ckpt ckpt/coach/coach_scw0_vocab.ckpt &> training_log/coach/baseline_scw0.log & 
# nohup python coach.py --cuda 0 -lr 5e-4 --n_sample 0 --tgt_dm SearchScreeningEvent --freeze_emb --tr --epoch 300 --emb_dim 400 --dropout 0.3 --hidden_size 200 --num_rnn_layer 2 --emb_file data/snips/cache/revised/slu+slot_word_char.npy --slot_emb_file data/snips/cache/revised/slot_word_char.dict --model_ckpt ckpt/coach/coach_sse0.ckpt --vocab_ckpt ckpt/coach/coach_sse0_vocab.ckpt &> training_log/coach/baseline_sse0.log & 