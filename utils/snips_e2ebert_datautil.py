import os

import scipy as sp
from utils.vocab import BERTVocab, Vocab, MultiVocab
import torch
import collections
from collections import defaultdict
import random
import pickle
import numpy as np
from data.snips.generate_slu_emb import slot_list
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import pickle as pkl


domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", 
              "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]


class Instance(object):
    def __init__(self, tokens, slu_tags=None, domain=None):
        self.tokens = tokens
        self.slu_tags = slu_tags
        self.domain = domain

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

# slot_list = ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist', 'object_name', 'object_type']
# slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 
#              'movie_name', 'location_name', 'restaurant_name', 'track', 
#              'restaurant_type', 'object_part_of_series_type', 'country', 'service', 
#              'poi', 'party_size_description', 'served_dish', 'genre', 
#              'current_location', 'object_select', 'album', 'object_name', 
#              'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 
#              'artist', 'cuisine', 'entity_name', 'object_type', 
#              'playlist_owner', 'timeRange', 'city', 'rating_value', 
#              'best_rating', 'rating_unit', 'year', 'party_size_number', 
#              'condition_description', 'condition_temperature']


slot2desp = {'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 
             'facility': 'facility', 'movie_name': 'movie name', 'location_name': 'location name', 
             'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 
             'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 
             'poi': 'position', 'party_size_description': 'person', 'served_dish': 'served dish', 
             'genre': 'genre', 'current_location': 'current location', 'object_select': 'object select', 
             'album': 'album', 'object_name': 'object name', 'state': 'location', 
             'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 
             'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 
             'entity_name': 'entity name', 'object_type': 'object type', 'playlist_owner': 'owner', 
             'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 
             'best_rating': 'best rating', 'rating_unit': 'rating unit', 'year': 'year', 
             'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}

def read_insts(file_reader, dm):
    for line in file_reader:
        new_inst = {"tokens": [], "slu_tags": [], "domain": dm}
        line = line.strip().split('\t')
        new_inst["tokens"] = line[0].split()
        new_inst["slu_tags"] = [x.split('-')[0] + '-' + slot2desp[x.split('-')[1]] if x != 'O' else x for x in line[1].split()]
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


def create_vocab(all_data_sets, bert_path):
    binary_vocab = Vocab(unk=None, bos=None, eos=None)
    slot_vocab = Vocab(pad='[PAD]', unk=None, bos=None, eos=None)
    slot_bio_vocab = Vocab(unk=None, bos=None, eos=None)
    for domain in domain_set:
        dataset = all_data_sets[f"{domain}"]
        for inst in dataset:
            binary_vocab.add([x.split('-')[0] if x != 'O' else x for x in inst.slu_tags])
            slot_vocab.add([x.split('-')[1] for x in inst.slu_tags if x != 'O'])
            slot_bio_vocab.add([x for x in inst.slu_tags])

    return MultiVocab(dict(
        bert=BERTVocab(bert_path),
        binary=binary_vocab,
        slot=slot_vocab,
        slot_bio=slot_bio_vocab
    ))


def batch_variable(batch_data, Vocab, mode='train'):
    # assert slot2desp.keys() == slot_list
    bsz = len(batch_data)
    token_max_len = max(len(inst.tokens) for inst in batch_data)
    token_seq_len = [len(inst.tokens) for inst in batch_data]
    
    bert_vocab = Vocab['bert']
    binary_vocab = Vocab['binary']
    slot_vocab = Vocab['slot']
    slot_bio_vocab = Vocab['slot_bio']
    
    tokens_list = []
    pref_chars = [x[0] for x in slot_vocab]

    token_mask = torch.zeros((bsz, token_max_len), dtype=torch.bool)
    bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    
    for i in range(bsz):
        tokens, slu_tags = batch_data[i].tokens, batch_data[i].slu_tags
        
        # convert token2id offsets.
        tokens_list.append(pref_chars + tokens)
        
        token_mask[i, :token_seq_len[i]].fill_(1)
        bio, slu = [], []
        for lbl in slu_tags:
            if lbl == 'O':
                bio.append(lbl)
                slu.append(slot_vocab.PAD)
            else:
                bio.append(lbl.split('-')[0])
                slu.append(lbl.split('-')[1])
        bio_label[i, :token_seq_len[i]] = torch.tensor([binary_vocab.inst2idx(x) for x in bio], dtype=torch.long)
        slu_label[i, :token_seq_len[i]] = torch.tensor([slot_vocab.inst2idx(x) for x in slu], dtype=torch.long)
        slu_bio_label[i, :token_seq_len[i]] = torch.tensor([slot_bio_vocab.inst2idx(x) for x in slu_tags], dtype=torch.long)

    bert_inputs = bert_vocab.batch_bertwd2id(tokens_list)
    
    return Batch(bert_inputs=bert_inputs,
                 
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
