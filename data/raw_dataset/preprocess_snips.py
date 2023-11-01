import os

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}


def read_data():
    domains, seqs_in, seqs_out = [], [], []
    ## domains
    # train
    with open("data/raw_dataset/original_snips_data/train/label", "r") as label_tr_f:
        for i, line in enumerate(label_tr_f):
            line = line.strip()
            domains.append(line)
    # valid
    with open("data/raw_dataset/original_snips_data/valid/label", "r") as label_val_f:
        for i, line in enumerate(label_val_f):
            line = line.strip()
            domains.append(line)
    # test
    with open("data/raw_dataset/original_snips_data/test/label", "r") as label_test_f:
        for i, line in enumerate(label_test_f):
            line = line.strip()
            domains.append(line)

    ## seqs_in
    # train
    with open("data/raw_dataset/original_snips_data/train/seq.in", "r") as seqin_tr_f:
        for i, line in enumerate(seqin_tr_f):
            line = line.strip()
            seqs_in.append(line)
    # valid
    with open("data/raw_dataset/original_snips_data/valid/seq.in", "r") as seqin_val_f:
        for i, line in enumerate(seqin_val_f):
            line = line.strip()
            seqs_in.append(line)
    # test
    with open("data/raw_dataset/original_snips_data/test/seq.in", "r") as seqin_test_f:
        for i, line in enumerate(seqin_test_f):
            line = line.strip()
            seqs_in.append(line)

    ## seqs_out
    # train
    with open("data/raw_dataset/original_snips_data/train/seq.out", "r") as seqout_tr_f:
        for i, line in enumerate(seqout_tr_f):
            line = line.strip()
            seqs_out.append(line)
    # valid
    with open("data/raw_dataset/original_snips_data/valid/seq.out", "r") as seqout_val_f:
        for i, line in enumerate(seqout_val_f):
            line = line.strip()
            seqs_out.append(line)
    # test
    with open("data/raw_dataset/original_snips_data/test/seq.out", "r") as seqout_test_f:
        for i, line in enumerate(seqout_test_f):
            line = line.strip()
            seqs_out.append(line)
    
    return domains, seqs_in, seqs_out

def preprocess_slu():
    domains, seqs_in, seqs_out = read_data()
    assert len(domains) == len(seqs_in) == len(seqs_out)
    
    for domain in list(domain2slot.keys()):
        os.mkdir(f'data/snips/{domain}')
    AddToPlaylistFile = open("data/snips/AddToPlaylist/AddToPlaylist.txt", "w")
    BookRestaurantFile = open("data/snips/BookRestaurant/BookRestaurant.txt", "w")
    GetWeatherFile = open("data/snips/GetWeather/GetWeather.txt", "w")
    PlayMusicFile = open("data/snips/PlayMusic/PlayMusic.txt", "w")
    RateBookFile = open("data/snips/RateBook/RateBook.txt", "w")
    SearchCreativeWorkFile = open("data/snips/SearchCreativeWork/SearchCreativeWork.txt", "w")
    SearchScreeningEventFile = open("data/snips/SearchScreeningEvent/SearchScreeningEvent.txt", "w")

    for domain, seq_in, seq_out in zip(domains, seqs_in, seqs_out):
        if domain == "AddToPlaylist":
            AddToPlaylistFile.write(seq_in + "\t" + seq_out + "\n")
        elif domain == "BookRestaurant":
            BookRestaurantFile.write(seq_in + "\t" + seq_out + "\n")
        elif domain == "GetWeather":
            GetWeatherFile.write(seq_in + "\t" + seq_out + "\n")
        elif domain == "PlayMusic":
            PlayMusicFile.write(seq_in + "\t" + seq_out + "\n")
        elif domain == "RateBook":
            RateBookFile.write(seq_in + "\t" + seq_out + "\n")
        elif domain == "SearchCreativeWork":
            SearchCreativeWorkFile.write(seq_in + "\t" + seq_out + "\n")
        elif domain == "SearchScreeningEvent":
            SearchScreeningEventFile.write(seq_in + "\t" + seq_out + "\n")

    AddToPlaylistFile.close()
    BookRestaurantFile.close()
    GetWeatherFile.close()
    PlayMusicFile.close()
    RateBookFile.close()
    SearchCreativeWorkFile.close()
    SearchScreeningEventFile.close()


def generate_seen_and_unseen_data():
    domains = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

    domain_to_all_seen_slots = {}
    for tgt_dm in domains:
        slots = []
        for src_dm in domains:
            if src_dm != tgt_dm:
                slots.extend(domain2slot[src_dm])
        slots = list(set(slots))  # remove repetition
        domain_to_all_seen_slots[tgt_dm] = slots

    for tgt_dm in domains:
        tgt_data = []
        with open(f'data/snips/{tgt_dm}/{tgt_dm}.txt') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                tgt_data.append({"utterance": line[0].strip().split(), "labels": line[1].strip().split()})
        seen_cnt, unseen_cnt = 0, 0

        seen_slots_file = open("data/snips/"+tgt_dm+"/seen_slots.txt", "w")
        unseen_slots_file = open("data/snips/"+tgt_dm+"/unseen_slots.txt", "w")
        tgt_seen_slots = domain_to_all_seen_slots[tgt_dm]

        # dm_data["utter"][500:] are for the test set
        for data in tgt_data[500:]:
            utter, labels = data['utterance'], data['labels']
            flag = False
            for item in labels:
                if "-" in item:
                    item = item.split("-")[1]
                    if item not in tgt_seen_slots:
                        flag = True
                        break
            if flag:
                unseen_cnt += 1
                # the utterance has at least one unseen slot
                unseen_slots_file.write(" ".join(utter) + "\t" + " ".join(labels) + "\n")
            else:
                seen_cnt += 1
                seen_slots_file.write(" ".join(utter) + "\t" + " ".join(labels) + "\n")
        
        print("Domain: %s; Seen samples: %d; Unseen samples: %d; Total samples: %d" % (tgt_dm, seen_cnt, unseen_cnt, unseen_cnt+seen_cnt))
        
        # Domain: AddToPlaylist; Seen samples: 480; Unseen samples: 1062; Total samples: 1542
        # Domain: BookRestaurant; Seen samples: 40; Unseen samples: 1533; Total samples: 1573
        # Domain: GetWeather; Seen samples: 623; Unseen samples: 977; Total samples: 1600
        # Domain: PlayMusic; Seen samples: 386; Unseen samples: 1214; Total samples: 1600
        # Domain: RateBook; Seen samples: 0; Unseen samples: 1556; Total samples: 1556
        # Domain: SearchCreativeWork; Seen samples: 1554; Unseen samples: 0; Total samples: 1554
        # Domain: SearchScreeningEvent; Seen samples: 168; Unseen samples: 1391; Total samples: 1559

if __name__ == "__main__":
    preprocess_slu()
    generate_seen_and_unseen_data()