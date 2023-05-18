import os
from utils.vocab import Vocab, MultiVocab
import torch
import collections
import pickle
import numpy as np

class Instance(object):
    def __init__(self, tokens, binary_tags:None, slu_tags=None, domain=None, template_list: list=None):
        self.tokens = tokens
        self.slu_tags = slu_tags
        self.binary_tags = binary_tags
        self.template_list = template_list
        self.domain = domain

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

y1_set = ["O", "B-Entity", "I-Entity"]
y2_set = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

entity_label_to_descrip = {
    "LOC": "location", 
    "PER": "person", 
    "ORG": "organization", 
    "MISC": "miscellaneous"}
entity_types = ["location", "person", "organization", "miscellaneous"]

def read_insts(file_reader, dm, use_label_encoder=False):
    tokens = []
    slu_tags = []
    binary_tags = []
    templates = [[], [], []]
    for line in file_reader:
        line = line.strip().split()
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            if use_label_encoder:
                num_of_O = sum([x != 'O' for x in binary_tags])
                if num_of_O == 0:
                    tokens, slu_tags, binary_tags, templates = [], [], [], [[], [], []]
                    continue
            assert len(templates[0]) != 0 and len(templates[1]) != 0 and len(templates[2]) != 0
            yield Instance(**{"tokens": tokens, "slu_tags": slu_tags, 
                "binary_tags": binary_tags, "template_list": templates, 
                "domain": dm})
            tokens, slu_tags, binary_tags, templates = [], [], [], [[], [], []]
            continue
        elif len(line) == 2:
            line.insert(0, ' ')

        assert len(line) == 3
        tokens.append(line[0].lower())
        binary_tags.append(line[1].split('-')[0])
        slu_tags.append(line[2])

        # template list
        if line[2][0] == 'O':
            templates[0].append(line[0].lower())
            templates[1].append(line[0].lower())
            templates[2].append(line[0].lower())
        elif line[2][0] == 'I':
            continue
        elif line[2][0] == 'B':
            templates[0].append(entity_label_to_descrip[line[2].split('-')[1]])
            for j in range(1, 3):
                idx = 0
                if entity_types[idx] != entity_label_to_descrip[line[2].split('-')[1]]:
                    templates[j].append(entity_types[idx % len(entity_types)])
                else:
                    idx += 1
                    templates[j].append(entity_types[idx % len(entity_types)])
                idx += 1


def load_data(domain, path, use_label_encoder=False):
    assert os.path.exists(path)
    dataset = []

    too_long = 0
    with open(path, 'r', encoding='utf-8') as fr:
        for inst in read_insts(fr, domain, use_label_encoder):
            if len(inst.tokens) < 512:
                dataset.append(inst)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return dataset


def create_vocab(all_data_sets):
    token_vocab = Vocab()
    slot_vocab = Vocab(unk=None, bos=None, eos=None)
    binary_vocab = Vocab(unk=None, bos=None, eos=None)
    domain_vocab = Vocab(unk=None, bos=None, eos=None)
    
    for inst in all_data_sets:
        token_vocab.add(inst.tokens)
        # token_vocab.add(inst.template_list[0])
        # token_vocab.add(inst.template_list[1])
        # token_vocab.add(inst.template_list[2])
        slot_vocab.add(inst.slu_tags)
        binary_vocab.add(inst.binary_tags)
        domain_vocab.add(inst.domain)

    print([x for x, _ in slot_vocab])
    print([x for x, _ in binary_vocab])
    print([x for x, _ in domain_vocab])

    return MultiVocab(dict(
        token=token_vocab,
        slot=slot_vocab,
        binary=binary_vocab,
        domain=domain_vocab
    ))


def _is_chinese(a_chr):
    return u'\u4e00' <= a_chr <= u'\u9fff'


def batch_variable(batch_data, Vocab, mode='train'):
    bsz = len(batch_data)
    token_max_len = max(len(inst.tokens) for inst in batch_data)
    token_seq_len = [len(inst.tokens) for inst in batch_data]
    template_max_len = max(len(template) for inst in batch_data for template in inst.template_list)
    template_seq_len = [len(template) for inst in batch_data for template in inst.template_list]
    
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
        tokens, template_list, binary_tags, slu_tags, domain = batch_data[i].tokens, batch_data[i].template_list, batch_data[i].binary_tags, batch_data[i].slu_tags, batch_data[i].domain
        token_ids[i, :token_seq_len[i]] = torch.tensor([token_vocab.inst2idx(x) for x in tokens], dtype=torch.long)
        token_mask[i, :token_seq_len[i]] = torch.tensor([1. for _ in range(token_seq_len[i])])
        
        template_ids[3*i, :template_seq_len[3*i]] = torch.tensor([token_vocab.inst2idx(x) for x in template_list[0]], dtype=torch.long)
        template_ids[3*i+1, :template_seq_len[3*i+1]] = torch.tensor([token_vocab.inst2idx(x) for x in template_list[1]], dtype=torch.long)
        template_ids[3*i+2, :template_seq_len[3*i+2]] = torch.tensor([token_vocab.inst2idx(x) for x in template_list[2]], dtype=torch.long)
        template_mask[3*i, :template_seq_len[3*i]] = torch.tensor([1. for _ in range(template_seq_len[3*i])])
        template_mask[3*i+1, :template_seq_len[3*i+1]] = torch.tensor([1. for _ in range(template_seq_len[3*i+1])])
        template_mask[3*i+2, :template_seq_len[3*i+2]] = torch.tensor([1. for _ in range(template_seq_len[3*i+2])])

        
        domain_ids[i] = domain_vocab.inst2idx(domain)
        bio_label[i, :token_seq_len[i]] = torch.tensor([binary_vocab.inst2idx(x) for x in binary_tags], dtype=torch.long)
        slu_label[i, :token_seq_len[i]] = torch.tensor([slu_vocab.inst2idx(x) for x in slu_tags], dtype=torch.long)
    
    return Batch(tokens=token_ids,
                 templates=template_ids,
                 domains=domain_ids,
                 
                 bio_label=bio_label,
                 slu_label=slu_label,
                 
                 token_mask=token_mask,
                 template_mask=template_mask)


class Batch:
    def __init__(self, **args):
        for prop, v in args.items():
            setattr(self, prop, v)

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, prop, val.to(device))
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
