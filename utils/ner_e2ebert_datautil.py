import os
from utils.vocab import BERTVocab, Vocab, MultiVocab
import torch
import collections
import pickle
import numpy as np
import json
import random
import copy
import numpy as np

domain_set = ['conll2003', 'tech']


class Instance(object):
    def __init__(self, tokens, slu_tags=None, domain=None):
        self.tokens = tokens
        self.slu_tags = slu_tags
        self.domain = domain

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


# domain to slot (different slots in different domain, one slot may appear in different domaina.)
domain2slot = {
    "conll2003": ['[PAD]', 'PER', 'ORG', 'LOC', 'MISC'],
    "tech": ['[PAD]', 'PER', 'ORG', 'LOC', 'MISC']
}

slot2desp = {'[PAD]': '[PAD]', 'PER': 'person', 'ORG': 'organization', 'LOC': 'location', 'MISC': 'miscellaneous'}
# slot2desp = {'[PAD]': '[PAD]', 'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC', 'MISC': 'MISC'}
domain2unseen = {'conll2003': [], 'tech': []}

def read_insts(file_reader, dm):
    new_inst = {"tokens": [], "slu_tags": [], "domain": dm}
    for line in file_reader:
        line = line.strip().split()
        if len(line) == 0:
            new_inst['domain'] = dm
            if len(new_inst['tokens']) > 0:
                yield(Instance(**new_inst))
            new_inst = {"tokens": [], "slu_tags": [], "domain": dm}
            continue
        assert len(line) == 3
        new_inst["tokens"].append(line[0])
        if line[2][0] == 'O':
            new_inst["slu_tags"].append(line[2][0])
        else:
            new_inst["slu_tags"].append(line[2][0:2] + slot2desp[line[2][2:]])
    
    if len(new_inst['tokens']) > 0:
        yield(Instance(**new_inst))


def load_data(path, domain):
    dataset = []
    assert os.path.exists(path)
    too_long = 0
    with open(path, 'r', encoding='utf-8') as fr:
        for inst in read_insts(fr, domain):
            if len(inst.tokens) < 512:
                dataset.append(inst)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return dataset


def create_vocab(all_data_sets, bert_path):
    binary_vocab = Vocab(unk=None, bos=None, eos=None)
    slot_bio_vocab = Vocab(pad='[PAD]', unk=None, bos=None, eos=None)
    for domain in domain_set:
        dataset = all_data_sets[f"{domain}"]
        for inst in dataset:
            binary_vocab.add([x[0] for x in inst.slu_tags])
            slot_bio_vocab.add([x for x in inst.slu_tags])

    return MultiVocab(dict(
        bert=BERTVocab(bert_path),
        binary=binary_vocab,
        slot_bio=slot_bio_vocab
    )) 


class SubstitutionBatch(object):
    def __init__(self, batch_data):
        self.batch = copy.deepcopy(batch_data)
    
    def substitute(self, i, tokens, slu_tags):
        self.batch[i].tokens = tokens
        self.batch[i].slu_tags = slu_tags


def batch_variable(batch_data, Vocab, tgt_dm, mask_p=(0., 0), mode='train'):
    # assert slot2desp.keys() == slot_list
    bsz = len(batch_data)
    desp2slot = {v: k for k, v in slot2desp.items()}
    
    # TODO: random entity substitution
    # with open('data/ner/conll2003/conll2003_ent4otherdm.json') as f:
    #     type2entity = json.load(f)

    # sub_batch = SubstitutionBatch(batch_data)
    # for i in range(bsz):
    #     tokens, slu_tags = batch_data[i].tokens, batch_data[i].slu_tags
    #     # # TODO: substitute the whole entity.
    #     if mode == 'train' and mask_p[0] > 0.:
    #         masked_tokens = []
    #         masked_slu_tags = []
    #         sub_flag = False
    #         for j, tok in enumerate(tokens):
    #             if slu_tags[j][0] == 'B' and np.random.rand() < mask_p[0]:
    #                 random.shuffle(type2entity[desp2slot[slu_tags[j][2:]]])
    #                 random_ent = type2entity[desp2slot[slu_tags[j][2:]]][0].split()
    #                 masked_tokens.extend(random_ent)
    #                 masked_slu_tags.append(slu_tags[j])
    #                 masked_slu_tags.extend([f'I-{slu_tags[j][2:]}'] * (len(random_ent)-1))
    #                 sub_flag = True
    #             elif slu_tags[j][0] == 'I':
    #                 if sub_flag:
    #                     continue
    #                 else:
    #                     masked_tokens.append(tok)
    #                     masked_slu_tags.append(slu_tags[j])
    #             else:
    #                 masked_tokens.append(tok)
    #                 masked_slu_tags.append(slu_tags[j])
    #                 sub_flag = False
    #     else:
    #         masked_tokens = tokens
    #         masked_slu_tags = slu_tags
        
    #     assert len(masked_tokens) == len(masked_slu_tags)
    #     sub_batch.substitute(i, masked_tokens, masked_slu_tags)
    
    # TODO: entity mask.
    # sub_batch = SubstitutionBatch(batch_data)
    # dm_num_type = len(domain2slot[batch_data[0].domain]) - 1
    # masked_type = slot2desp[domain2slot[batch_data[0].domain][1:][mask_p[1] % dm_num_type]]
    # for i in range(bsz):
    #     tokens, slu_tags = batch_data[i].tokens, batch_data[i].slu_tags
        # # TODO: mask the whole entity.
        # masked_tokens = []
        # masked_tags = []
        # if mode == 'train' and np.random.rand() < mask_p[0]:
        #     for tok, lbl in zip(tokens, slu_tags):
        #         if lbl[0] == 'B' and lbl[2:] == masked_type:
        #             masked_tokens.append('[MASK]')
        #             masked_tags.append(lbl)
        #         else:
        #             if lbl[0] == 'I' and lbl[2:] == masked_type:
        #                 continue
        #             else:
        #                 masked_tokens.append(tok)
        #                 masked_tags.append(lbl)
        # else:
        #     masked_tokens = tokens
        #     masked_tags = slu_tags
        # sub_batch.substitute(i, masked_tokens, masked_tags)

    # TODO: mask entity
    token_max_len = max(len(inst.tokens) for inst in batch_data)
    token_seq_len = [len(inst.tokens) for inst in batch_data]
    
    bert_vocab = Vocab['bert']
    binary_vocab = Vocab['binary']
    slot_bio_vocab = Vocab['slot_bio']
    
    tokens_list = []

    token_mask = torch.zeros((bsz, token_max_len), dtype=torch.bool)
    bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    
    # 0 is [PAD]
    # dm_num_type = len(domain2slot[batch_data[0].domain]) - 1
    # masked_type = slot2desp[domain2slot[batch_data[0].domain][1:][mask_p[1] % dm_num_type]]
    for i in range(bsz):
        # TODO: mask entity
        tokens, slu_tags = batch_data[i].tokens, batch_data[i].slu_tags
        pref_chars = [slot2desp[x] for x in domain2slot[batch_data[i].domain]]
        
        # TODO: entity_mask (for training only)
        # if mode == 'train' and mask_p[0] > 0.:
        #     # TODO: bernouli sampler
        #     bernouli_mask = np.random.binomial(1, mask_p[0], len(tokens))
        #     masked_tokens = ['[MASK]' if _label[0] != 'O' and _label[2:] == masked_type and bernouli_mask[i] == 1 else _tok for i, (_tok, _label) in enumerate(zip(tokens, slu_tags))]
        #     # masked_tokens = ['[MASK]' if _label[0] != 'O' and np.random.rand() < mask_p else _tok for _tok, _label in zip(tokens, slu_tags)]
        # else:
        #     masked_tokens = tokens
        
        tokens_list.append(pref_chars + tokens)
        
        token_mask[i, :token_seq_len[i]].fill_(1)
        bio, slu = [], []
        for lbl in slu_tags:
            if lbl == 'O':
                bio.append(lbl)
                slu.append('[PAD]')
            else:
                bio.append(lbl[0])
                slu.append(lbl.split('-')[1])
        
        bio_label[i, :token_seq_len[i]] = torch.tensor([binary_vocab.inst2idx(x) for x in bio], dtype=torch.long)
        # TODO: mask entity
        slu_label[i, :token_seq_len[i]] = torch.tensor([domain2slot[batch_data[i].domain].index(desp2slot[x]) for x in slu], dtype=torch.long)
        slu_bio_label[i, :token_seq_len[i]] = torch.tensor([slot_bio_vocab.inst2idx(x) for x in slu_tags], dtype=torch.long)
    
    bert_inputs = bert_vocab.batch_bertwd2id(tokens_list)
    unseen_inputs = None
        
    return Batch(bert_inputs=bert_inputs,
                 unseen_inputs=unseen_inputs,
                 
                 bio_label=bio_label,
                 slu_label=slu_label,
                 slu_bio_label=slu_bio_label,
                 
                 token_mask=token_mask)


class Batch:
    def __init__(self, **args):
        for prop, v in args.items():
            setattr(self, prop, v)

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, prop, val.to(device))
            elif isinstance(val, dict):
                val_ = {}
                for k, v in val.items():
                    val_[k] = v.to(device)
                setattr(self, prop, val_)
            elif isinstance(val, collections.abc.Sequence) or isinstance(val, collections.abc.Iterable):
                val_ = [v.to(device) if torch.is_tensor(v) else v for v in val]
                setattr(self, prop, val_)
        return self


def save_to(path, obj):
    if os.path.exists(path):
        return None
    with open(path, 'wb') as fw:
        pickle.dump(obj, fw)
    print('Obj saved!')


def load_from(pkl_file):
    with open(pkl_file, 'rb') as fr:
        obj = pickle.load(fr)
    return obj
