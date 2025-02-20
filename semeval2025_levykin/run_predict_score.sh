#!/bin/bash

python3 ./predict.py \
    --prompt-file prompts.txt \
    --prompt-num 0 \
    --retrieval-file /home/admin/projects/semeval2025/rag/en_val_rag_summary.jsonl \
    --pred-file /home/admin/projects/semeval2025/predictions/predictions.jsonl \
    --temperature 1.0 \
    --prompt-params "doc_1,doc_2,doc_3,doc_4,doc_5,model_input,model_output_text"

python3 ./scorer.py \
    --ref-file /home/admin/projects/semeval2025/data/mushroom.en-val.v2.jsonl \
    --pred-file /home/admin/projects/semeval2025/predictions/predictions.jsonl \
    --output-file /home/admin/projects/semeval2025/scores.txt \
    --comment "inferense: prompt #1 temperature:1.0"
