import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.e2e_bert_tagger import End2endSLUTagger
from config.slu_conf import args_config
from utils.dataset import DataLoader
from utils.snips_e2ebert_datautil import load_data, create_vocab, batch_variable, save_to, load_from
# from data.snips.generate_slu_emb import slot_list
from utils.conlleval import evaluate
from logger.logger import logger
from torch.nn.utils.rnn import pad_sequence
import pickle as pkl


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        print(self.args)
        
        self.all_dataset = load_data()
        self.tgt_dm = args.tgt_dm
        self.n_sample = args.n_sample
        
        self.test_set = []
        # split train/dev set.
        for dm, insts in self.all_dataset.items():
            if dm == self.tgt_dm:
                self.test_set.extend(insts[500:])

        print('test data size:', len(self.test_set))
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size, shuffle=False)

        self.vocabs = load_from(args.vocab_ckpt)
        print([x for x in self.vocabs["binary"]])
        print([x for x in self.vocabs["slot"]])
        
        self.slu_model = End2endSLUTagger(
            self.vocabs,
            bert_embed_dim=768,
            hidden_size=self.args.hidden_size,
            num_type=len(self.vocabs['slot']),
            num_bound=len(self.vocabs['binary']),
            num_bert_layer=4,
            dropout=args.dropout,
            bert_model_path='bert_model/bert-base-uncased'
        )

        self.restore_states(args.model_ckpt)
        self.slu_model = self.slu_model.to(args.device)

        print(self.slu_model)
        total_params = sum([p.numel() for gen in [self.slu_model.parameters()] for p in gen if p.requires_grad])
        print("Training %dM trainable parameters..." % (total_params/1e6))

    def save_states(self, save_path, best_test_metric=None):
        self.slu_model.zero_grad()
        check_point = {'slu_model_state': self.slu_model.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'args': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')

    def restore_states(self, load_path):
        ckpt = torch.load(load_path)
        # torch.set_rng_state(ckpt['rng_state'])
        self.slu_model.load_state_dict(ckpt['slu_model_state'])
        # self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.slu_model.zero_grad()
        # self.optimizer.zero_grad()
        # self.model.train()
        logger.info('Loading the previous model states ...')
        # print('Previous best prf result is: %s' % ckpt['best_prf'])


    def evaluate(self):
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        # # TODO: bad case analysis
        # wrong_samples = []
        # wrong_pred = []
        # wrong_gold = []
        wrong_pred = []
        correct_pred = []
        with torch.no_grad():
            for i, batcher in enumerate(self.test_loader):
                batch = batch_variable(batcher, self.vocabs, mode='test')
                batch.to_device(self.args.device)
                
                batch_bio_pred, batch_slu_pred, bert_repr = self.slu_model.evaluate(batch.bert_inputs, batch.token_mask)
                bert_repr = F.normalize(bert_repr, p=2, dim=-1)
                seq_len = batch.token_mask.sum(dim=-1)
                
                # 记录batch内每个pred slot label
                for j, (bio_pred, slu_pred) in enumerate(zip(batch_bio_pred, batch_slu_pred)):
                    bio_gold_tags.extend([x.split('-')[0] if x != 'O' else x for x in batcher[j].slu_tags])
                    slu_gold_tags.extend(batcher[j].slu_tags)

                    sent_bio_pred = self.vocabs['binary'].idx2inst(bio_pred[:seq_len[j]].tolist())
                    sent_slu_pred = self.vocabs['slot'].idx2inst(slu_pred[:seq_len[j]].tolist())
                    sent_slu_pred = [x+'-'+y if x != 'O' else 'O' for x, y in zip(sent_bio_pred, sent_slu_pred)]
                    bio_pred_tags.extend(sent_bio_pred)
                    slu_pred_tags.extend(sent_slu_pred)
                    # TODO: save embeddings of good/bad case.
                    if i == 1 and j == 12:
                        for word_c, (tok, gold, pred) in enumerate(zip(batcher[j].tokens, batcher[j].slu_tags, sent_slu_pred)):
                            if pred[2:] != gold[2:] and gold[0] != 'O' and pred[0] != 'O':
                                wrong_pred.append((tok, gold[2:], pred[2:], bert_repr[j, len(self.vocabs['slot'])+1+word_c]))
                            elif pred[2:] == gold[2:] and gold[0] != 'O' and pred[0] != 'O':
                                correct_pred.append((tok, gold[2:], pred[2:], bert_repr[j, len(self.vocabs['slot'])+1+word_c]))
                # TODO: save TSNE embeddings:
                if i == 1:
                    # assert len(wrong_pred) > 1 and len(correct_pred) > 1
                    with open('/home/tjuwlz/syj_project/zero-shot-slu/ckpt/tsne_emb/sample.pkl', 'wb') as f:
                        pkl.dump({"wrong": wrong_pred, "correct": correct_pred}, f)
                    np.save(f'ckpt/tsne_emb/{self.args.tgt_dm}.npy', bert_repr[12, 1: len(self.vocabs['slot'])+1].cpu().detach().numpy())

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

    set_seeds(1314)
    evaluator = Evaluator(args)
    print(evaluator.evaluate())