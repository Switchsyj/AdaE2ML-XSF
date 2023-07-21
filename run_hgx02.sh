#!/bin/bash
#SBATCH --job-name=bash # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=8 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pot # number of gpus per node

#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/%x-%j.err # output and error log file names (%x for job id)

# python ner_e2e_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.5 --model_ckpt testlog/test_tech.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/backbone_ner.log
# python ner_e2e_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.5 --cl --cl_temperature 0.1 --cl_type cosine --model_ckpt testlog/test_tech.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/backbone_cl_ner.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.3 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/wandb_bert_domain_cl_br0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/wandb_bert_domain_br0.log

# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_orth_cl_atp0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_orth_cl_br0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_orth_cl_gw0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_orth_cl_pm0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_orth_cl_rb0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --cl --cl_type cosine --cl_temperature 1.0 --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_orth_cl_scw0.log
# python slu_e2e_bert_domain_unseen_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --cl --cl_type cosine --cl_temperature 0.05 --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_orth_cl_sse0.log

nohup python slu_e2e_bert_f2train.py --cuda 0 -lr 1e-3 --n_sample 50 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_atp.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50cl_atp0.log &
nohup python slu_e2e_bert_f2train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.07 --model_ckpt /cognitive_comp/shiyuanjun/zero-shot-slu/ckpt/end2end_cl/bert_domain_f2brcl.ckpt --vocab_ckpt /cognitive_comp/shiyuanjun/zero-shot-slu/ckpt/vocab/f2brcl_vocab.ckpt &> fs_log/testbrcl.log &
nohup python slu_e2e_bert_f2train.py --cuda 0 -lr 1e-3 --n_sample 50 --tgt_dm GetWeather --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_gw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50cl_gw0.log &
nohup python slu_e2e_bert_f2train.py --cuda 1 -lr 1e-3 --n_sample 50 --tgt_dm PlayMusic --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt testlog/test_pm.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50cl_pm0.log &
nohup python slu_e2e_bert_f2train.py --cuda 1 -lr 1e-3 --n_sample 50 --tgt_dm RateBook --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt testlog/test_rb.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50cl_rb0.log &
nohup python slu_e2e_bert_f2train.py --cuda 1 -lr 1e-3 --n_sample 50 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_scw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50cl_scw0.log &
nohup python slu_e2e_bert_f2train.py --cuda 2 -lr 1e-3 --n_sample 50 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_sse.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50cl_sse0.log &

nohup python slu_e2e_bert_f2train.py --cuda 2 -lr 1e-3 --n_sample 50 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.1 --model_ckpt testlog/test_atp.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50_atp0.log &
nohup python slu_e2e_bert_f2train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.1 --model_ckpt /cognitive_comp/shiyuanjun/zero-shot-slu/ckpt/end2end_cl/bert_domain_f2br.ckpt --vocab_ckpt /cognitive_comp/shiyuanjun/zero-shot-slu/ckpt/vocab/f2br_vocab.ckpt &> fs_log/testbr.log &
nohup python slu_e2e_bert_f2train.py --cuda 3 -lr 1e-3 --n_sample 50 --tgt_dm GetWeather --epoch 30 --dropout 0.1 --model_ckpt testlog/test_gw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50_gw0.log &
nohup python slu_e2e_bert_f2train.py --cuda 3 -lr 1e-3 --n_sample 50 --tgt_dm PlayMusic --epoch 30 --dropout 0.1 --model_ckpt testlog/test_pm.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50_pm0.log &
nohup python slu_e2e_bert_f2train.py --cuda 3 -lr 1e-3 --n_sample 50 --tgt_dm RateBook --epoch 30 --dropout 0.1 --model_ckpt testlog/test_rb.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50_rb0.log &
nohup python slu_e2e_bert_f2train.py --cuda 2 -lr 1e-3 --n_sample 50 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.1 --model_ckpt testlog/test_scw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50_scw0.log &
nohup python slu_e2e_bert_f2train.py --cuda 1 -lr 1e-3 --n_sample 50 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.1 --model_ckpt testlog/test_sse.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> fs_log/bert_domain_adalabel_fs50_sse0.log &


nohup python ner_e2e_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.5 --model_ckpt testlog/ner.ckpt --vocab_ckpt testlog/ner_vocab.ckpt &> testlog/bert_domain_adalabel_nerdrop05.log &
nohup python ner_e2e_train.py --cuda 7 -lr 1e-3 --n_sample 0 --tgt_dm tech --epoch 30 --dropout 0.5 --cl --cl_type cosine --cl_temperature 0.05 --model_ckpt testlog/nercl.ckpt --vocab_ckpt testlog/nercl_vocab.ckpt &> testlog/bert_domain_adalabel_nercl005.log &


# nohup python slu_mbt_train.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_mbtrev4_detwosh_atp0.log &
# nohup python slu_mbt_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_mbtrev4_detwosh_br0.log &
# nohup python slu_mbt_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_mbtrev4_detwosh_gw0.log &
# nohup python slu_mbt_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_mbtrev4_detwosh_pm0.log &
# nohup python slu_mbt_train.py --cuda 7 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_mbtrev4_detwosh_rb0.log &
# nohup python slu_mbt_train.py --cuda 7 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_mbtrev4_detwosh_scw0.log &
# nohup python slu_mbt_train.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEv1ent --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_mbtrev4_detwosh_sse0.log &

# nohup python slu2step_train.py --cuda 4 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.1 --model_ckpt testlog/test_atp.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_freezecrflstm_bioall_lora_atp0.log &
# nohup python slu2step_train.py --cuda 4 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.1 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_freezecrflstm_bioall_lora_br0.log &
# nohup python slu2step_train.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.1 --model_ckpt testlog/test_gw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_freezecrflstm_bioall_lora_gw0.log &
# nohup python slu2step_train.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.1 --model_ckpt testlog/test_pm.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_freezecrflstm_bioall_lora_pm0.log &
# nohup python slu2step_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.1 --model_ckpt testlog/test_rb.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_freezecrflstm_bioall_lora_rb0.log &
# nohup python slu2step_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.1 --model_ckpt testlog/test_scw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_freezecrflstm_bioall_lora_scw0.log &
# nohup python slu2step_train.py --cuda 3 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.1 --model_ckpt testlog/test_sse.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_freezecrflstm_bioall_lora_sse0.log &

nohup python slu_e2e_bert_domain_unseen_train.py --cuda 4 -lr 1e-3 --n_sample 0 --tgt_dm AddToPlaylist --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_atp.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_atp0.log &
nohup python slu_e2e_bert_domain_unseen_train.py --cuda 4 -lr 1e-3 --n_sample 0 --tgt_dm BookRestaurant --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.3 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_br0.log &
nohup python slu_e2e_bert_domain_unseen_train.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm GetWeather --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt testlog/test_gw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_gw0.log &
nohup python slu_e2e_bert_domain_unseen_train.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm PlayMusic --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt testlog/test_pm.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_pm0.log &
nohup python slu_e2e_bert_domain_unseen_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm RateBook --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_rb.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_rb0.log &
nohup python slu_e2e_bert_domain_unseen_train.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm SearchCreativeWork --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.7 --model_ckpt testlog/test_scw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_scw0.log &
nohup python slu_e2e_bert_domain_unseen_train.py --cuda 3 -lr 1e-3 --n_sample 0 --tgt_dm SearchScreeningEvent --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.05 --model_ckpt testlog/test_sse.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_cl_sse0.log &

# export LAUNCHER="torchrun \
#     --nproc_per_node 8 \
#     example.py \
#     --ckpt_dir /cognitive_comp/liuyibo/llm/llama/65B \
#     --tokenizer_path /cognitive_comp/liuyibo/llm/llama/tokenizer.model \
#     "

nohup python -u gpt3DataConstruction.py &> data_const6k.log &

# cross dataset experiment.
nohup python slu_e2e_bert_f2xdataset.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm searching --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_atp.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_searching0cl.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm ordering --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_br.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_ordering0cl.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm comparing --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_gw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_comparing0cl.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 6 -lr 1e-3 --n_sample 0 --tgt_dm payment --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_pm.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_payment0cl.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 7 -lr 1e-3 --n_sample 0 --tgt_dm customer-service --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_rb.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_customer0cl.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 7 -lr 1e-3 --n_sample 0 --tgt_dm delivery --epoch 30 --dropout 0.1 --cl --cl_type cosine --cl_temperature 0.5 --model_ckpt testlog/test_scw.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_delivery0cl.log &

nohup python slu_e2e_bert_f2xdataset.py --cuda 0 -lr 1e-3 --n_sample 0 --tgt_dm searching --epoch 30 --dropout 0.1 --model_ckpt testlog/test_search.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_searching0.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 1 -lr 1e-3 --n_sample 0 --tgt_dm ordering --epoch 30 --dropout 0.1 --model_ckpt testlog/test_order.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_ordering0.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 2 -lr 1e-3 --n_sample 0 --tgt_dm comparing --epoch 30 --dropout 0.1 --model_ckpt testlog/test_compare.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_comparing0.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 3 -lr 1e-3 --n_sample 0 --tgt_dm payment --epoch 30 --dropout 0.1 --model_ckpt testlog/test_payment.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_payment0.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 4 -lr 1e-3 --n_sample 0 --tgt_dm customer-service --epoch 30 --dropout 0.1 --model_ckpt testlog/test_customer.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_customer0.log &
nohup python slu_e2e_bert_f2xdataset.py --cuda 5 -lr 1e-3 --n_sample 0 --tgt_dm delivery --epoch 30 --dropout 0.1 --model_ckpt testlog/test_delivery.ckpt --vocab_ckpt testlog/test_vocab.ckpt &> testlog/bert_domain_adalabel_delivery0.log &