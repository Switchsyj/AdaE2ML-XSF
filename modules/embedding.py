import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


class PretrainedEmbedding(nn.Module):
    def __init__(self, vocab, emb_file, num_words, emb_dim, dropout=0.):
        super(PretrainedEmbedding, self).__init__()
        self.emb_file = emb_file
        self.num_words = num_words
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.padding_idx = vocab.inst2idx('<pad>')
        
        self.embedding = nn.Embedding(self.num_words, self.emb_dim, padding_idx=self.padding_idx)
        
        print(f"load pretrained embedding from {self.emb_file}")
        if emb_file.endswith('pkl'):
            with open(emb_file, 'rb') as f:
                word_emb_dict = pickle.load(f)
            _embedding = np.zeros((len(vocab._inst2idx), emb_dim))
            for word, id in vocab._inst2idx.items():
                if word in word_emb_dict:
                    _embedding[id] = torch.tensor(word_emb_dict[word])
                else:
                    raise Exception
        elif emb_file.endswith('npy'):
            _embedding = np.load(emb_file)
        elif emb_file.endswith('txt'):
            nb_embeddings = 0
            _embedding = None
            with open(emb_file, 'r') as f:
                for line in f:
                    tokens = line.strip().split(' ')
                    if len(tokens) < 10:
                        continue

                    wd = tokens[0]
                    idx =vocab._inst2idx.get(wd)
                    if idx is not None:
                        vec = np.asarray(tokens[1:], dtype=np.float32)
                        if _embedding is None:
                            vec_size = len(vec)
                            _embedding = np.random.uniform(-0.5/vec_size, 0.5/vec_size, (self.num_words, vec_size))
                            _embedding[self.padding_idx] = np.zeros(vec_size, np.float32)
                        _embedding[idx] = vec
                        nb_embeddings += 1
        _embedding /= np.std(_embedding)
        
        self.embedding.weight.data.copy_(torch.tensor(_embedding, dtype=torch.float32))
        print(f'OOV rate: {nb_embeddings / self.num_words}')
        
    def forward(self, input):
        return self.embedding(input)
        