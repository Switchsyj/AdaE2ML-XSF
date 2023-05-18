#!/bin/bash
#SBATCH --job-name=code # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=8 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pot # number of gpus per node

#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/%x-%j.err # output and error log file names (%x for job id)

# python ner_e2e_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.5 --model_ckpt testlog/test_tech.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/backbone_ner.log
# python ner_e2e_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.5 --cl --cl_temperature 0.1 --cl_type cosine --model_ckpt testlog/test_tech.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/backbone_cl_ner.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/wandb_bert_domain_cl_br0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/wandb_bert_domain_br0.log

# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_atp0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_br0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_gw0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_pm0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_rb0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_scw0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_sse0.log

# export LAUNCHER="torchrun \
#     --nproc_per_node 8 \
#     example.py \
#     --ckpt_dir /cognitive_comp/liuyibo/llm/llama/65B \
#     --tokenizer_path /cognitive_comp/liuyibo/llm/llama/tokenizer.model \
#     "
