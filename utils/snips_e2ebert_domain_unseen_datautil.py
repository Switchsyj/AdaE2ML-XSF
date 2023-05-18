import os
from utils.vocab import BERTVocab, Vocab, MultiVocab
import torch
import collections
from collections import defaultdict
import pickle
import numpy as np
import random
import json


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


# domain to slot (different slots in different domain, one slot may appear in different domaina.)
domain2slot = {
    "AddToPlaylist": ['[PAD]', 'music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['[PAD]', 'city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['[PAD]', 'city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['[PAD]', 'genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['[PAD]', 'object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['[PAD]', 'object_name', 'object_type'],
    "SearchScreeningEvent": ['[PAD]', 'timeRange', 'movie_type', 'object_location_type', 'object_type', 'location_name', 'spatial_relation', 'movie_name']
}

domain2unseen = {
    "AddToPlaylist": ['playlist_owner', 'entity_name'],
    "BookRestaurant": ['facility', 'restaurant_name', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'party_size_description'],
    "GetWeather": ['current_location', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'service', 'year', 'album', 'track'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": [],
    "SearchScreeningEvent": ['movie_type', 'object_location_type', 'location_name', 'movie_name']
}

# slot2desp = {'[PAD]': '[PAD]', 'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 
#              'facility': 'facility', 'movie_name': 'movie name', 'location_name': 'location name', 
#              'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 
#              'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 
#              'poi': 'position', 'party_size_description': 'person', 'served_dish': 'served dish', 
#              'genre': 'genre', 'current_location': 'current location', 'object_select': 'this current', 
#              'album': 'album', 'object_name': 'object name', 'state': 'location', 
#              'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 
#              'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 
#              'entity_name': 'entity name', 'object_type': 'object type', 'playlist_owner': 'owner', 
#              'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 
#              'best_rating': 'best rating', 'rating_unit': 'rating unit', 'year': 'year', 
#              'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}

slot2desp = {'[PAD]': '[PAD]', 'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 
            'facility': 'facility', 'movie_name': 'movie name', 'location_name': 'location name', 
            'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 
            'object_part_of_series_type': 'series type', 'country': 'country', 'service': 'service', 
            'poi': 'position', 'party_size_description': 'party size description', 'served_dish': 'served dish', 
            'genre': 'genre', 'current_location': 'current location', 'object_select': 'object select', 
            'album': 'album', 'object_name': 'object name', 'state': 'state', 
            'sort': 'sort', 'object_location_type': 'object location type', 'movie_type': 'movie type', 
            'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 
            'entity_name': 'entity name', 'object_type': 'object type', 'playlist_owner': 'playlist owner', 
            'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 
            'best_rating': 'best rating', 'rating_unit': 'rating unit', 'year': 'year', 
            'party_size_number': 'party size number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}


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
    slot_bio_vocab = Vocab(pad='[PAD]', unk=None, bos=None, eos=None)
    for domain in domain_set:
        dataset = all_data_sets[f"{domain}"]
        for inst in dataset:
            binary_vocab.add([x.split('-')[0] if x != 'O' else x for x in inst.slu_tags])
            slot_bio_vocab.add([x for x in inst.slu_tags])

    return MultiVocab(dict(
        bert=BERTVocab(bert_path),
        binary=binary_vocab,
        slot_bio=slot_bio_vocab
    ))


def batch_variable(batch_data, Vocab, tgt_dm, mode='train'):
    # assert slot2desp.keys() == slot_list
    bsz = len(batch_data)
    desp2slot = {v: k for k, v in slot2desp.items()}
    
    # TODO: random entity substitution
    # with open(f'data/snips/{tgt_dm}/{tgt_dm}_ent4otherdm.json') as f:
    #     type2entity = json.load(f)

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
    #     batch_data[i].tokens = masked_tokens
    #     batch_data[i].slu_tags = masked_slu_tags
    
    token_max_len = max(len(inst.tokens) for inst in batch_data)
    token_seq_len = [len(inst.tokens) for inst in batch_data]
    
    bert_vocab = Vocab['bert']
    binary_vocab = Vocab['binary']
    slot_bio_vocab = Vocab['slot_bio']
    
    tokens_list = []
    # pref_char_list = []

    token_mask = torch.zeros((bsz, token_max_len), dtype=torch.bool)
    bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    
    # TODO: uttr_label CL
    # cl_label = torch.zeros((bsz, token_max_len + len(domain2slot[batch_data[0].domain])), dtype=torch.long)
    
    # dm_num_type = len(domain2slot[batch_data[0].domain]) - 1
    # masked_type = slot2desp[domain2slot[batch_data[0].domain][1:][mask_p[1] % dm_num_type]]
    for i in range(bsz):
        tokens, slu_tags = batch_data[i].tokens, batch_data[i].slu_tags 
        pref_chars = [slot2desp[x] for x in domain2slot[batch_data[i].domain]]

        # TODO: entity token mask (for training only)
        # if mode == 'train' and mask_p > 0.:
        #     # TODO: bernouli sampler
        #     bernouli_mask = np.random.binomial(1, mask_p, len(tokens))
        #     # masked_tokens = ['[MASK]' if _label[0] != 'O' and _label[2:] == masked_type and bernouli_mask[i] == 1 else _tok for i, (_tok, _label) in enumerate(zip(tokens, slu_tags))]
        #     masked_tokens = ['[ENT]' if _label[0] != 'O' and bernouli_mask[i] == 1 else _tok for i, (_tok, _label) in enumerate(zip(tokens, slu_tags))]
        #     # masked_tokens = ['[MASK]' if _label[0] != 'O' and np.random.rand() < mask_p[1] else _tok for _tok, _label in zip(tokens, slu_tags)]
        # else:
        #     masked_tokens = tokens
        
        # tokens_list.append(['In', 'domain', f'{batch_data[i].domain}:'] + pref_chars + tokens)
        # pref_char_list.append(pref_chars)
        tokens_list.append(pref_chars + tokens)
        
        token_mask[i, :token_seq_len[i]].fill_(1)
        bio, slu = [], []
        for lbl in slu_tags:
            if lbl == 'O':
                bio.append(lbl)
                slu.append('[PAD]')
            else:
                bio.append(lbl.split('-')[0])
                slu.append(lbl.split('-')[1])
        
        bio_label[i, :token_seq_len[i]] = torch.tensor([binary_vocab.inst2idx(x) for x in bio], dtype=torch.long)
        slu_label[i, :token_seq_len[i]] = torch.tensor([domain2slot[batch_data[i].domain].index(desp2slot[x]) for x in slu], dtype=torch.long)
        # TODO: uttr+label CL
        # cl_label[i, :len(domain2slot[batch_data[i].domain])] = torch.tensor([domain2slot[batch_data[i].domain].index(desp2slot[x]) for x in pref_chars], dtype=torch.long)
        # cl_label[i, len(domain2slot[batch_data[i].domain]): token_seq_len[i]+len(domain2slot[batch_data[i].domain])] = torch.tensor([domain2slot[batch_data[i].domain].index(desp2slot[x]) for x in slu], dtype=torch.long)
        

        slu_bio_label[i, :token_seq_len[i]] = torch.tensor([slot_bio_vocab.inst2idx(x) for x in slu_tags], dtype=torch.long)
    
    bert_inputs = bert_vocab.batch_bertwd2id(tokens_list)

    return Batch(bert_inputs=bert_inputs,
                 
                 bio_label=bio_label,
                 slu_label=slu_label,
                #  cl_label=cl_label,
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
