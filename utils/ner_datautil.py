import os
from utils.vocab import BERTVocab, Vocab, MultiVocab
import torch
import collections
import pickle
from data.snips.generate_slu_emb import slot_list


class Instance(object):
    def __init__(self, tokens, ner_tags=None, domain=None):
        self.tokens = tokens
        self.ner_tags = ner_tags
        self.domain = domain

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


slot2desp = {'PER': 'person', 'LOC': 'location', 'ORG': 'organization', 'MISC': 'miscellaneous'}

def read_insts(file_reader, dm):
    tokens = []
    ner_tags = []
    for line in file_reader:
        line = line.strip().split()
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            yield Instance(**{"tokens": tokens, "ner_tags": ner_tags, "domain": dm})
            tokens, ner_tags = [], []
            continue
        elif len(line) == 2:
            line.insert(0, ' ')

        assert len(line) == 3
        tokens.append(line[0])
        if line[2][0] == 'O':
            ner_tags.append('O')
        else:
            ner_tags.append(line[2])


def load_data(dm, path):
    dataset = []
    assert os.path.exists(path)
    too_long = 0
    with open(path, 'r', encoding='utf-8') as fr:
        for inst in read_insts(fr, dm):
            if len(inst.tokens) < 512:
                dataset.append(inst)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return dataset


def create_vocab(all_datasets, bert_path):
    slot_vocab = Vocab(pad='[PAD]', unk=None, bos=None, eos=None)
    for inst in all_datasets:
        slot_vocab.add(inst.ner_tags)

    return MultiVocab(dict(
        bert=BERTVocab(bert_path),
        slot=slot_vocab,
    ))


def batch_variable(batch_data, Vocab, mode='train'):
    # assert slot2desp.keys() == slot_list
    bsz = len(batch_data)
    token_max_len = max(len(inst.tokens) for inst in batch_data)
    token_seq_len = [len(inst.tokens) for inst in batch_data]
    
    bert_vocab = Vocab['bert']
    slot_vocab = Vocab['slot']
    
    tokens_list = []

    token_mask = torch.zeros((bsz, token_max_len), dtype=torch.bool)
    slu_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    
    for i in range(bsz):
        tokens, ner_tags = batch_data[i].tokens, batch_data[i].ner_tags
        
        # convert token2id offsets.
        tokens_list.append(tokens)
        
        token_mask[i, :token_seq_len[i]].fill_(1)
        slu_label[i, :token_seq_len[i]] = torch.tensor([slot_vocab.inst2idx(x) for x in ner_tags], dtype=torch.long)

    bert_inputs = bert_vocab.batch_bertwd2id(tokens_list)
    
    return Batch(bert_inputs=bert_inputs,
                 slu_label=slu_label,         
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
