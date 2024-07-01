export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# /home/zhengbaj/tir5/zhiqing/hf_home/hub/models--allenai--tulu-2-dpo-70b/snapshots/0ab5c875f0070d5aee8d36bc55f41de440a13f02
# allenai/tulu-2-dpo-70b

# python -u generate_honesty_score.py \
#     --model_name_or_path "/home/zhengbaj/tir5/zhiqing/hf_home/hub/models--allenai--tulu-2-dpo-70b/snapshots/0ab5c875f0070d5aee8d36bc55f41de440a13f02" \
#     --tokenizer_name "allenai/tulu-2-dpo-70b" \
#     --model_tag "tulu-2" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity_hard.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard_scored.jsonl" \
#     --layer_start 30 \
#     --layer_end 50 \
#     --start_answer_token "istant"

# mean: 0.7723698989142643
# min: 0.809434668663422
# prompt: 0.7285660801198053

# /data/models/huggingface/meta-llama/Llama-2-7b-chat-hf

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --model_tag "llama-2-7b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity_hard.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard_scored.jsonl" \
#     --layer_start 10 \
#     --layer_end 20 \
#     --start_answer_token "INST"

# mean: 0.6450767502807937
# min: 0.6334706102583302
# prompt: 0.5728191688506178

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --model_tag "llama-2-13b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity_hard.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard_scored.jsonl" \
#     --layer_start 20 \
#     --layer_end 30 \
#     --start_answer_token "INST"

# mean: 0.6836390864844627
# min: 0.6802695619618121
# prompt: 0.585174092100337

python -u generate_honesty_score.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
    --model_tag "llama-2-70b" \
    --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
    --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity_hard.json" \
    --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard.json" \
    --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_hard_scored.jsonl" \
    --layer_start 20 \
    --layer_end 40 \
    --layer_step 2 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --start_answer_token "INST"

# mean: 0.6956196181205541
# min: 0.6731561213028828
# prompt: 0.5964058405091726
