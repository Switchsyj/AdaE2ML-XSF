import os
from utils.vocab import BERTVocab, Vocab, MultiVocab
import torch
import collections
from collections import defaultdict
import pickle
import jsonlines as jsl
import pdb
from copy import deepcopy as deepcopy


class Instance(object):
    def __init__(self, tokens, slu_tags=None, domain=None):
        self.tokens = tokens
        self.slu_tags = slu_tags
        self.domain = domain

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
    

slot2desp_textbook = {'O': '[PAD]', '[PAD]': '[PAD]', 'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 
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
            'party_size_number': 'party size number', 'condition_description': 'weather', 'condition_temperature': 'temperature',
            'fromloc.city_name': 'fromloc city name', 'toloc.city_name': 'toloc city name', 'round_trip': 'round trip', 
            'arrive_date.day_number': 'arrive date day number', 'arrive_date.month_name': 'arrive date month name', 'stoploc.city_name': 'stoploc city name', 
            'arrive_time.time_relative': 'arrive time time relative', 'arrive_time.time': 'arrive time time', 'meal_description': 'meal description', 
            'meal_code': 'meal code', 'restriction_code': 'restriction code', 'days_code': 'days code', 'booking_class': 'booking class', 'transport_type': 'transport type', 'state_code': 'state code', 'state_name': 'state name', 'time': 'time', 'today_relative': 'today relative', 'time_relative': 'time relative', 'month_name': 'month name', 'day_number': 'day number',
            'depart_date.day_number': 'depart date day number', 'depart_date.month_name': 'depart date month name', 
            'depart_time.period_of_day': 'depart time period of day', 'airline_name': 'airline name', 'toloc.state_name': 'toloc state name', 
            'depart_time.time_relative': 'depart time time relative', 'depart_time.time': 'depart time time', 'depart_date.day_name': 'depart date day name', 
            'depart_date.date_relative': 'depart date date relative', 'or': 'or', 'class_type': 'class type', 
            'fromloc.airport_name': 'fromloc airport name', 'meal': 'meal', 'flight_mod': 'flight mod', 'economy': 'economy', 
            'city_name': 'city name', 'airline_code': 'airline code', 'depart_date.today_relative': 'depart date today relative', 
            'flight_stop': 'flight stop', 'toloc.state_code': 'toloc state code', 'fromloc.state_name': 'fromloc state name', 
            'toloc.airport_name': 'toloc airport name', 'connect': 'connect', 'arrive_date.day_name': 'arrive date day name', 
            'fromloc.state_code': 'fromloc state code', 'arrive_date.today_relative': 'arrive date today relative', 'depart_date.year': 'depart date year', 
            'depart_time.start_time': 'depart time start time', 'depart_time.end_time': 'depart time end time', 'arrive_time.start_time': 'arrive time start time', 
            'arrive_time.end_time': 'arrive time end time', 'cost_relative': 'cost relative', 'flight_days': 'flight days', 'mod': 'mod', 'airport_name': 'airport name', 
            'aircraft_code': 'aircraft code', 'toloc.country_name': 'toloc country name', 'toloc.airport_code': 'toloc airport code', 
            'return_date.date_relative': 'return date date relative', 'flight_number': 'flight number', 'fromloc.airport_code': 'fromloc airport code', 
            'arrive_time.period_of_day': 'arrive time period of day', 'depart_time.period_mod': 'depart time period mod', 'flight_time': 'flight time', 
            'return_date.day_name': 'return date day name', 'fare_amount': 'fare amount', 'arrive_date.date_relative': 'arrive date date relative', 'arrive_time.period_mod': 'arrive time period mod', 
            'period_of_day': 'period of day', 'stoploc.state_code': 'stoploc state code', 'fare_basis_code': 'fare basis code', 'stoploc.airport_name': 'stoploc airport name', 
            'return_time.period_of_day': 'return time period of day', 'return_time.period_mod': 'return time period mod', 'return_date.today_relative': 'return date today relative', 
            'return_date.month_name': 'return date month name', 'return_date.day_number': 'return date day number', 'compartment': 'compartment', 'day_name': 'day name', 'airport_code': 'airport code', 
            'stoploc.airport_code': 'stoploc airport code', 'flight': 'flight',
            'product_name': 'product name', 'category': 'category', 'price': 'price',
            'color': 'color', 'material': 'material', 'payment_type': 'payment type', 'inquiry_type': 'inquiry type',
            }


def load_data(src, tgt):
    all_dataset = {"train": defaultdict(list), "dev": defaultdict(list), "test": defaultdict(list)}
    too_long = 0
    empty_entity = 0
    # load all the data from source for training
    for src_split in ['train', 'dev', 'test']:
        with jsl.open(f"data/merge_dataset/{src}/{src_split}.jsonl", 'r') as f:
            for data in f:
                if sum([x == 'O' for x in data['slu_tags']]) < len(data['slu_tags']):
                    if len(data['tokens']) < 512:
                        all_dataset['train'][f"{data['domain']}"].append(Instance(**data))            
                    else:
                        too_long += 1
                        pdb.set_trace()
                else:
                    empty_entity += 1
            
    with jsl.open(f"data/merge_dataset/{tgt}/dev.jsonl", 'r') as f:
        cnt = 0
        for data in f:
            cnt += 1
            if len(data['tokens']) < 512:
                if cnt <= 500:
                    all_dataset['dev'][f"{data['domain']}"].append(Instance(**data))
                else:
                    all_dataset['test'][f"{data['domain']}"].append(Instance(**data))
            else:
                too_long += 1
                pdb.set_trace()
            
    with jsl.open(f"data/merge_dataset/{tgt}/test.jsonl", 'r') as f:
        for data in f:
            if len(data['tokens']) < 512:
                all_dataset['test'][f"{data['domain']}"].append(Instance(**data))
            else:
                too_long += 1
                pdb.set_trace()

    print(f'{too_long} sentences exceeds 512 tokens')
    print(f'{empty_entity} sentences with no entities')
     
    return all_dataset


def create_vocab(all_data_sets, bert_path):
    binary_vocab = Vocab(unk=None, bos=None, eos=None)
    slot_bio_vocab = Vocab(pad='[PAD]', unk=None, bos=None, eos=None)
    
    # merge train/dev/test set.
    for _split in ['train', 'dev', 'test']:
        domain_set = list(all_data_sets[_split].keys())
        for domain in domain_set:
            dataset = all_data_sets[_split][f"{domain}"]
            for inst in dataset:
                binary_vocab.add([x.split('-')[0] if x != 'O' else x for x in inst.slu_tags])
                slot_bio_vocab.add([x for x in inst.slu_tags])

    return MultiVocab(dict(
        bert=BERTVocab(bert_path),
        binary=binary_vocab,
        slot_bio=slot_bio_vocab
    ))


def batch_variable(batch_data, Vocab, slot2desp, domain2slot, mode='train'):
    bsz = len(batch_data)
     
    token_max_len = max(len(inst.tokens) for inst in batch_data)
    token_seq_len = [len(inst.tokens) for inst in batch_data]
    
    bert_vocab = Vocab['bert']
    binary_vocab = Vocab['binary']
    slot_bio_vocab = Vocab['slot_bio']
    
    tokens_list = []

    # zero-padding
    token_mask = torch.zeros((bsz, token_max_len), dtype=torch.bool)
    bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    slu_bio_label = torch.zeros((bsz, token_max_len), dtype=torch.long)
    
    for i in range(bsz):
        tokens, slu_tags = batch_data[i].tokens, batch_data[i].slu_tags
        domain_slots = [x for x in domain2slot[batch_data[i].domain]]
            
        pref_chars = [slot2desp[x] for x in domain_slots]
        tokens_list.append(pref_chars + tokens)
        
        token_mask[i, :token_seq_len[i]].fill_(1)
        bio, slu = [], []
        for lbl in slu_tags:
            if lbl == 'O':
                bio.append(lbl)
                slu.append('[PAD]')
            else:
                bio.append(lbl.split('-')[0])
                slu.append('-'.join(lbl.split('-')[1:]))
        
        bio_label[i, :token_seq_len[i]] = torch.tensor([binary_vocab.inst2idx(x) for x in bio], dtype=torch.long)
        slu_label[i, :token_seq_len[i]] = torch.tensor([domain_slots.index(x) for x in slu], dtype=torch.long)
        slu_bio_label[i, :token_seq_len[i]] = torch.tensor([slot_bio_vocab.inst2idx(x) for x in slu_tags], dtype=torch.long)
    
    bert_inputs = bert_vocab.batch_bertwd2id(tokens_list)

    return Batch(bert_inputs=bert_inputs,
                 bio_label=bio_label,
                 slu_label=slu_label,
                 slu_bio_label=slu_bio_label,    
                 token_mask=token_mask,
                 num_slot_type=torch.tensor([len(domain_slots)], dtype=torch.long))


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
