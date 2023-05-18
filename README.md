## How to run
### Configuration
- ```--tgt_dm:``` Target domain
- ```--n_samples:``` Number of samples used in the target domain, for K-shot, set n_samples to K.
- ```--model_ckpt:``` Saved model path
- ```--vocab_ckpt:``` Saved vocab path
- ```--cl:``` using slot-level cl
- ```--cl_type:``` slot-level cl metric function
- ```--cl_temperature:``` slot-level cl temperature \tau

### Cross-domain Slot Filling
Train our model for zero-shot adaptation to AddToPlaylist domain
```console
❱❱❱ python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_atp0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_atp0_vocab.ckpt
```

Train our model without CL for zero-shot adaptation to AddToPlaylist domain
```console
❱❱❱ python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.3 --model_ckpt ckpt/end2end_cl/bert_domain_atp0.ckpt --vocab_ckpt ckpt/vocab/bert_domain_atp0_vocab.ckpt
```

### Cross-domain NER
Train our model for zero-resource adaptation to sci-tech domain
```console
❱❱❱ python ner_e2e_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.5 --cl --cl_type cosine --cl_temperature 0.1 --model_ckpt testlog/test_tech.ckpt --vocab_ckpt testlog/test_vocab.ckpt
```

Train our model without slot-level CL for zero-resource adaptation to sci-tech domain
```console
❱❱❱ python ner_baseline.py --exp_name ct --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4
```