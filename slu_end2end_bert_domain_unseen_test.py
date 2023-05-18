import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from model.e2e_bert_domain_unseen_tagger import End2endSLUTagger
from config.slu_conf import args_config
from utils.dataset import DataLoader
from utils.snips_e2ebert_domain_unseen_datautil import load_data, batch_variable, load_from, domain2slot, slot2desp, domain2unseen, read_insts
from utils.conlleval import evaluate
from logger.logger import logger

def load_samples(path, dm):
    test_set = []
    assert os.path.exists(path)
    too_long = 0
    with open(path, 'r', encoding='utf-8') as fr:
        for inst in read_insts(fr, dm):
            if len(inst.tokens) < 512:
                test_set.append(inst)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return test_set


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        print(self.args)
        
        # self.all_dataset = load_data()
        self.tgt_dm = args.tgt_dm
        self.n_sample = args.n_sample
        
        self.test_set = []
        # # split train/dev set.
        # for dm, insts in self.all_dataset.items():
        #     if dm == self.tgt_dm:
        #         self.test_set.extend(insts[500:])

        # TODO: test on seen/unseen data
        self.test_set = load_samples(f"data/snips/{self.args.tgt_dm}/seen_slots.txt", f'{self.args.tgt_dm}')
        # TODO: exclude val set
        # self.test_set = self.test_set[500:]
        print('test data size:', len(self.test_set))
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size, shuffle=False)

        self.vocabs = load_from(args.vocab_ckpt)
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

        self.restore_states(args.model_ckpt)
        self.slu_model = self.slu_model.to(args.device)

    def save_states(self, save_path, best_test_metric=None):
        self.slu_model.zero_grad()
        check_point = {'slu_model_state': self.slu_model.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'args': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')

    def restore_states(self, load_path):
        ckpt = torch.load(load_path)
        self.slu_model.load_state_dict(ckpt['slu_model_state'])
        self.slu_model.zero_grad()
        logger.info('Loading the previous model states ...')
        # print('Previous best prf result is: %s' % ckpt['best_prf'])

    def evaluate(self):
        self.slu_model.eval()
        
        bio_pred_tags = []
        bio_gold_tags = []
        slu_pred_tags = []
        slu_gold_tags = []
        # TODO: bad case analysis
        # wrong_samples = []
        # wrong_pred = []
        # wrong_gold = []
        desp2slot = {v: k for k, v in slot2desp.items()}
        with torch.no_grad():
            for i, batcher in enumerate(self.test_loader):
                batch = batch_variable(batcher, self.vocabs, tgt_dm=self.args.tgt_dm, mode='test')
                batch.to_device(self.args.device)
                
                batch_bio_pred, batch_slu_pred = self.slu_model.evaluate(batch.bert_inputs, num_type=len(domain2slot[batcher[0].domain]), mask=batch.token_mask)
                # bert_repr = F.normalize(bert_repr, p=2, dim=-1)
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

                    # TODO: bad case analysis
        #             if sum([x != y for x, y in zip(sent_slu_pred, batcher[j].slu_tags)]) > 0:
        #                 wrong_samples.append(' '.join(batcher[j].tokens))
        #                 wrong_pred.append(' '.join(sent_slu_pred))
        #                 wrong_gold.append(' '.join([x.split('-')[0] + '-' + desp2slot[x.split('-')[1]] if x != 'O' else x for x in batcher[j].slu_tags]))

        # with open('data/snips/badcase_sse_cl.txt', 'a+') as f:
        #     for sent, pred, gold in zip(wrong_samples, wrong_pred, wrong_gold):
        #         f.write(sent + '\n')
        #         f.write(pred + '\n')
        #         f.write(gold + '\n\n')
                    
        assert len(slu_pred_tags) == len(slu_gold_tags)
        bio_p, bio_r, bio_f = evaluate([x + '-1' if x != 'O' else x for x in bio_gold_tags], 
                                       [x + '-1' if x != 'O' else x for x in bio_pred_tags], verbose=False)
        
        # examine seen tags
        # seen_gold = ['O' if x[0] == 'O' or x[2:] in domain2unseen[f'{self.args.tgt_dm}'] else x for x in slu_gold_tags]
        # seen_pred = ['O' if _g[0] != 'O' and _g[2:] in domain2unseen[f'{self.args.tgt_dm}'] else _p for _p, _g in zip(slu_pred_tags, slu_gold_tags)]
        # examine slot unseen tags
        # unseen_gold = ['O' if x[0] == 'O' or x[2:] not in domain2unseen[f'{self.args.tgt_dm}'] else x for x in slu_gold_tags]
        # unseen_pred = ['O' if _g[0] != 'O' and _g[2:] not in domain2unseen[f'{self.args.tgt_dm}'] else _p for _p, _g in zip(slu_pred_tags, slu_gold_tags)]
        
        # with open('data/snips/badcase_unseen_br_cl.txt', 'a+') as f:
        #     for sent, pred, gold in zip(wrong_samples, wrong_pred, wrong_gold):
        #         pred = ['O' if _g[0] != 'O' and _g[2:] not in domain2unseen[f'{self.args.tgt_dm}'] else _p for _p, _g in zip(pred.split(), gold.split())]
        #         gold = ['O' if x[0] == 'O' or x[2:] not in domain2unseen[f'{self.args.tgt_dm}'] else x for x in gold.split()]
        #         f.write(sent + '\n')
        #         f.write(' '.join(pred) + '\n')
        #         f.write(' '.join(gold) + '\n\n')
        
        # slu_p, slu_r, slu_f = evaluate(unseen_gold, unseen_pred, verbose=False)
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