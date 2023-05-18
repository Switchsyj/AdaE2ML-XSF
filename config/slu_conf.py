import os
import json
import argparse


def data_config(data_path):
    assert os.path.exists(data_path)
    with open(data_path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)
    print(opts)
    return opts


def args_config():
    parse = argparse.ArgumentParser('Semantic Dependency Parsing')

    parse.add_argument('--cuda', type=int, default=0, help='cuda device, default cpu')

    parse.add_argument('--bert_path', type=str, default="bert_model/bert-base-uncased", help='path to pretrained bert model')
    parse.add_argument('--bert_lr', type=float, default=2e-5, help='bert learning rate')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parse.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('--batch_size', type=int, default=32, help='batch size')
    parse.add_argument('--val_batch_size', type=int, default=32, help='validation batch size')
    parse.add_argument('--test_batch_size', type=int, default=32, help='test batch size')
    parse.add_argument('--n_sample', type=int, default=0, help='few shot sample')
    parse.add_argument('--tgt_dm', type=str, default="AddToPlaylist", help='target domain')
    parse.add_argument('--epoch', type=int, default=20, help='max epoch')
    parse.add_argument('--patient', type=int, default=5, help='patient')
    
    # modules
    parse.add_argument('--tr', default=False, action='store_true', help='use template regularization')
    parse.add_argument('--cl', default=False, action='store_true', help='use contrastive learning')
    parse.add_argument('--cl_temperature', type=float, default=1.0, help='contrastive learning temperature')
    parse.add_argument('--cl_type', type=str, default='cosine', help='CL loss function')
    parse.add_argument('--ft', default=False, action='store_true', help='two step finetuning on the target domain')
    
    parse.add_argument('--emb_file', type=str, default='data/snips/cache/slu_word_char_embs_with_slotembs.npy', help='pretrained word embedding file')
    parse.add_argument('--emb_dim', type=int, default=768, help='pretrained word embedding dim')
    parse.add_argument('--bio_emb_dim', type=int, default=10, help='BIO embedding dim')
    parse.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parse.add_argument('--hidden_size', type=int, default=200, help='hidden size for LSTM')
    parse.add_argument('--bidirectional', default=True, action='store_false', help='Bi-LSTM')
    parse.add_argument('--num_rnn_layer', type=int, default=1, help='num of LSTM layer')
    parse.add_argument('--slot_emb_file', type=str, default="data/snips/cache/slot_word_char_embs_based_on_each_domain.pkl", help='slot embedding file')
    parse.add_argument('--freeze_emb', default=False, action='store_true', help='freeze embedding')
    
    parse.add_argument('--model_ckpt', type=str, default="ckpt/best_model.ckpt", help='path to save the best model')
    parse.add_argument('--vocab_ckpt', type=str, default="ckpt/vocab.ckpt", help='path to save the vocab state')
    
    args = parse.parse_args()

    print(vars(args))

    return args
