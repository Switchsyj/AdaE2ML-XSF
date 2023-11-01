import jsonlines as jsl
from collections import defaultdict, Counter
import json
import copy
import os
import pdb


if __name__ == '__main__':
    data_split = ['train', 'dev', 'test']
    
    # preprocessing SNIPS.
    labels = defaultdict(Counter)
    limits = 100
    dataset = defaultdict(list)
    domain_cnt = defaultdict(int)
    for _split in data_split:

        with open(f"data/raw_dataset/original_snips_data/{_split}/seq.in", 'r', encoding='utf-8') as in_reader, \
            open(f"data/raw_dataset/original_snips_data/{_split}/seq.out", 'r', encoding='utf-8') as out_reader, \
            open(f"data/raw_dataset/original_snips_data/{_split}/label", 'r', encoding='utf-8') as domain_reader:
            for _in in in_reader:
                _out = out_reader.readline().strip()
                _domain = domain_reader.readline().strip().lower()
                
                new_inst = {"tokens": _in.strip().lower().split(), 
                            "slu_tags": [x.split('-')[0] + '-' + x.split('-')[1].lower() if x != 'O' else 'O' for x in _out.strip().split()], 
                            "domain": f"{_domain}"}
                domain_cnt[_domain] += 1
                labels[_domain].update([x.split('-')[1].lower() if x != 'O' else 'O' for x in _out.strip().split()])
                dataset[_split].append(new_inst)
        assert len(list(domain_cnt.keys())) == 7
    
    for _split in data_split:
        new_dataset = []
        for data in dataset[_split]:
            new_dataset.append(data)
        print(f"======writing {len(new_dataset)} to: data/merge_dataset/snips/{_split}.jsonl======")
        with jsl.open(f'data/merge_dataset/snips/{_split}.jsonl', 'w') as f:
            f.write_all(new_dataset)
    
    overall_num_labels = []
    for k, v in labels.items():
        overall_num_labels.extend(list(dict(v).keys()))
    overall_num_labels = list(set(overall_num_labels))
    
    assert len(labels) == 7
    label_dict = defaultdict(list)
    for k, v in labels.items():
        if len(v) > 0:
            label_dict[k].extend(list(dict(v).keys()))

    for k, v in label_dict.items():
        print(f"****** {len(v)} slot labels in {k} domain, {v} ******")
    
    with open(f'data/merge_dataset/snips/labels.json', 'w') as f:
        json.dump(label_dict, f)
    
    print(f"======Output {len(overall_num_labels)} labels for {len(label_dict)} domains splited by domain: data/merge_dataset/snips/labels.json======")

## ========= ATIS ===========
    # preprocessing ATIS.
    labels = defaultdict(Counter)
    merged_domain_keys = []
    limits = 100
    dataset = defaultdict(list)
    domain_cnt = defaultdict(int)
    for _split in data_split:   
        with open(f"data/raw_dataset/atis/{_split}.txt", 'r') as f:
            tokens = []
            slu_tags = []
            label_cnt = defaultdict(int)
            for line in f:
                line = line.strip().split(' ')
                if len(line) == 2:
                    tokens.append(line[0].lower())
                    _tag = line[1].split('-')[1].lower() if line[1] != 'O' else 'O'
                    label_cnt[_tag] += 1
                    slu_tags.append(line[1].strip().split('-')[0] + '-' + line[1].strip().split('-')[1].lower() if line[1] != 'O' else 'O')
                elif len(line) == 1 and line[0] != '':
                    dataset[_split].append({"tokens": copy.deepcopy(tokens), 
                                    "slu_tags": copy.deepcopy(slu_tags), 
                                    "domain": f"{line[0].lower().strip()}"})
                    tokens = []
                    slu_tags = []
                    domain_cnt[line[0].lower().strip()] += 1
                    labels[f"{line[0].lower().strip()}"].update(label_cnt)
                    label_cnt = defaultdict(int)
                else:
                    continue
    
    # merge data less than `number_limits` into `others`
    for _domain, v in domain_cnt.items():
        if v < limits:
            merged_domain_keys.append(_domain)
    
    for _split in data_split: 
        new_dataset = []
        for data in dataset[_split]:
            if data['domain'] in merged_domain_keys:
                data['domain'] = 'atis_others'
            new_dataset.append(data)
        print(f"======writing to: data/merge_dataset/atis/{_split}.jsonl with {len(new_dataset)} examples ======")
        with jsl.open(f'data/merge_dataset/atis/{_split}.jsonl', 'w') as f:
            f.write_all(new_dataset)
    
    overall_num_labels = []
    for k, v in labels.items():
        overall_num_labels.extend(list(dict(v).keys()))
    overall_num_labels = list(set(overall_num_labels))
    
    # label + 'O'
    assert len(overall_num_labels) == 84
    label_dict = defaultdict(list)
    for k, v in labels.items():
        if len(v) > 0:
            if k in merged_domain_keys:
                label_dict['atis_others'].extend(list(dict(v).keys()))
            else:
                label_dict[k].extend(list(dict(v).keys()))
    
    # filter overlap labels
    if len(merged_domain_keys) > 0:
        label_dict['atis_others'] = list(set(label_dict['atis_others']))
 
    for k, v in label_dict.items():
        print(f"****** {len(v)} slot labels in {k} domain, {v} ******")

    with open(f'data/merge_dataset/atis/labels.json', 'w') as f:
        json.dump(label_dict, f)
    
    print(f"======Output {len(overall_num_labels)} labels for {len(label_dict)} domains splited by domain: data/merge_dataset/atis/labels.json======")
