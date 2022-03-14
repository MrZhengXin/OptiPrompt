#!/bin/bash

OUTPUTS_DIR=lama
MODEL=bert-large-cased
RAND=none

for REL in P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937; 
do

    DIR=${OUTPUTS_DIR}/${REL}
    mkdir -p ${DIR}

    python3 code/run_optiprompt.py \
        --relation_profile relation_metainfo/LAMA_relations.jsonl \
        --relation ${REL} \
        --common_vocab_filename common_vocabs/common_vocab_cased.txt \
        --model_name ${MODEL} \
        --do_train \
        --train_data data/autoprompt_data/${REL}/train.jsonl \
        --dev_data data/autoprompt_data/${REL}/dev.jsonl \
        --do_eval \
        --test_data data/LAMA-TREx/${REL}.jsonl \
        --output_dir ${DIR} \
        --random_init ${RAND} \
        --init_manual_template \
        --output_predictions \
        --k 128

done

python3 code/accumulate_results.py ${OUTPUTS_DIR}




# wiki uni
OUTPUTS_DIR=wiki_uni
MODEL=bert-large-cased
RAND=none

for REL in P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937; 
do

    DIR=${OUTPUTS_DIR}/${REL}
    mkdir -p ${DIR}

    python3 code/run_optiprompt.py \
        --relation_profile relation_metainfo/LAMA_relations.jsonl \
        --relation ${REL} \
        --common_vocab_filename common_vocabs/common_vocab_cased.txt \
        --model_name ${MODEL} \
        --train_data data/autoprompt_data/${REL}/train.jsonl \
        --dev_data data/autoprompt_data/${REL}/dev.jsonl \
        --do_eval \
        --test_data data/wiki_uni/${REL}.jsonl \
        --output_dir ${DIR} \
        --random_init ${RAND} \
        --output_predictions 
done

mkdir -p wiki_uni/wiki_uni
mkdir -p lama/lama
python3 code/count_prediction.py
cp -r wiki_uni/wiki_uni ../LANKA_journal/data/opti_prompt_bert_data/distribution/lama_original
cp -r lama/lama ../LANKA_journal/data/opti_prompt_bert_data/distribution/lama_original







OUTPUTS_DIR=case_based_10_train
MODEL=bert-large-cased
RAND=none
i=0
N_GPU=4
for REL in P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937; 
do

    DIR=${OUTPUTS_DIR}/${REL}
    mkdir -p ${DIR}
    i=$(($((${i} + 1)) % ${N_GPU}))

    CUDA_VISIBLE_DEVICES=${i} python3 code/run_optiprompt.py \
        --relation_profile relation_metainfo/LAMA_relations.jsonl \
        --relation ${REL} \
        --common_vocab_filename common_vocabs/common_vocab_cased.txt \
        --model_name ${MODEL} \
        --train_data data/autoprompt_data/${REL}/train.jsonl \
        --dev_data data/autoprompt_data/${REL}/dev.jsonl \
        --do_eval \
        --test_data data/LAMA-TREx/${REL}.jsonl \
        --output_dir ${DIR} \
        --random_init ${RAND} \
        --init_manual_template \
        --output_predictions \
        --few_shot_count 10 \
        --k 128 &
done & wait

python3 code/accumulate_results.py ${OUTPUTS_DIR} > ${OUTPUTS_DIR}/res.txt
python3 code/analyze_intype_rank.py > ${OUTPUTS_DIR}/table_figure_6.txt
python3 code/analyze_type_precision.py > ${OUTPUTS_DIR}/table_4.txt




