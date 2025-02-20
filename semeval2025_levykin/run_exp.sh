#!/bin/bash

python3 ./exp.py \
    --prompt-file prompts.txt \
    --retrieval-file /home/admin/projects/levykin/semeval2025_levykin/rag/en_val_rag.jsonl \
    --pred-folder /home/admin/projects/levykin/semeval2025_levykin/predictions/ \
    --ref-file /home/admin/projects/levykin/semeval2025_levykin/data/mushroom.en-val.v2.jsonl \
    --scores-file /home/admin/projects/levykin/semeval2025_levykin/scores.jsonl \
    --temperatures "0.1" \
    --prompt-params "doc_1,doc_2,doc_3,doc_4,doc_5,model_input,model_output_text" \
    --exp-name "my_exp"
