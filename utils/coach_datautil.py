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

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", 
              "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 
             'movie_name', 'location_name', 'restaurant_name', 'track', 
             'restaurant_type', 'object_part_of_series_type', 'country', 'service', 
             'poi', 'party_size_description', 'served_dish', 'genre', 
             'current_location', 'object_select', 'album', 'object_name', 
             'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 
             'artist', 'cuisine', 'entity_name', 'object_type', 
             'playlist_owner', 'timeRange', 'city', 'rating_value', 
             'best_rating', 'rating_unit', 'year', 'party_size_number', 
             'condition_description', 'condition_temperature']

def read_insts(file_reader, dm):
    for line in file_reader:
        new_inst = {"tokens": [], "slu_tags": [], "binary_tags": [], "template_list": [], "domain": dm}
        line = line.strip().split('\t')
        new_inst["tokens"] = line[0].split()
        new_inst["slu_tags"] = line[1].split()
        new_inst["binary_tags"] = [x[0] for x in line[1].split()]
        
        templates = [[], [], []]
        """
            template_each_sample[0] is correct template
            template_each_sample[1] and template_each_sample[2] are incorrect template (replace correct slots with other slots)
        """
        assert len(line[0].split()) == len(line[1].split())
        for token, slu_tag in zip(line[0].split(), line[1].split()):
            if slu_tag == 'O':
                templates[0].append(token)
                templates[1].append(token)
                templates[2].append(token)
            elif slu_tag[0] == 'I':
                continue
            elif slu_tag[0] == 'B':
                templates[0].append(slu_tag.split('-')[1])
                slot_list_copy = slot_list.copy()
                np.random.shuffle(slot_list_copy)
                idx = 0
                for j in range(1, 3):
                    if slot_list_copy[idx] != slu_tag.split('-')[1]:
                        templates[j].append(slot_list_copy[idx])
                    else:
                        idx += 1
                        templates[j].append(slot_list_copy[idx])
                    idx += 1
            else:
                raise Exception
        new_inst["template_list"] = templates
        yield(Instance(**new_inst))


def load_data():
    all_dataset = {}
    for domain in domain_set:
        path = f"data/snips/{domain}/{domain}.txt"
        assert os.path.exists(path)
        
        all_dataset[f"{domain}"] = []
        too_long = 0
        with open(path, 'r', encoding='utf-8') as fr:
            for inst in read_insts(fr, domain):
                if len(inst.tokens) < 512:
                    all_dataset[f"{domain}"].append(inst)
                else:
                    too_long += 1
        print(f'{too_long} sentences exceeds 512 tokens')
    return all_dataset


def create_vocab(all_data_sets, use_label_encoder=False):
    token_vocab = Vocab(pad='PAD', unk='UNK', bos=None, eos=None)
    slot_vocab = Vocab(unk=None, bos=None, eos=None)
    binary_vocab = Vocab(unk=None, bos=None, eos=None)
    domain_vocab = Vocab(pad=None, unk=None, bos=None, eos=None)
    for domain in domain_set:
        dataset = all_data_sets[f"{domain}"]
        for inst in dataset:
            token_vocab.add(inst.tokens)
            slot_vocab.add(inst.slu_tags)
            binary_vocab.add(inst.binary_tags)
            domain_vocab.add(inst.domain)
    if use_label_encoder:
        token_vocab.add(slot_list)
    # print([x for x, _ in slot_vocab])
    print([x for x, _ in binary_vocab])
    # print([x for x, _ in domain_vocab])

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
    
    bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long).fill_(binary_vocab.inst2idx('O'))
    slu_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    token_mask = torch.zeros((bsz, token_max_len), dtype=torch.bool)
    template_mask = torch.zeros((bsz*3, template_max_len), dtype=torch.float)
    for i in range(bsz):
        tokens, template_list, binary_tags, slu_tags, domain = batch_data[i].tokens, batch_data[i].template_list, batch_data[i].binary_tags, batch_data[i].slu_tags, batch_data[i].domain
        token_ids[i, :token_seq_len[i]] = torch.tensor(token_vocab.inst2idx(tokens), dtype=torch.long)
        token_mask[i, :token_seq_len[i]] = torch.tensor([1. for _ in range(token_seq_len[i])])
        
        template_ids[3*i, :template_seq_len[3*i]] = torch.tensor(token_vocab.inst2idx(template_list[0]), dtype=torch.long)
        template_ids[3*i+1, :template_seq_len[3*i+1]] = torch.tensor(token_vocab.inst2idx(template_list[1]), dtype=torch.long)
        template_ids[3*i+2, :template_seq_len[3*i+2]] = torch.tensor(token_vocab.inst2idx(template_list[2]), dtype=torch.long)
        template_mask[3*i, :template_seq_len[3*i]] = torch.tensor([1. for _ in range(template_seq_len[3*i])])
        template_mask[3*i+1, :template_seq_len[3*i+1]] = torch.tensor([1. for _ in range(template_seq_len[3*i+1])])
        template_mask[3*i+2, :template_seq_len[3*i+2]] = torch.tensor([1. for _ in range(template_seq_len[3*i+2])])

        
        domain_ids[i] = domain_vocab.inst2idx(domain)
        bio_label[i, :token_seq_len[i]] = torch.tensor(binary_vocab.inst2idx(binary_tags), dtype=torch.long)
        slu_label[i, :token_seq_len[i]] = torch.tensor(slu_vocab.inst2idx(slu_tags), dtype=torch.long)
    
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
