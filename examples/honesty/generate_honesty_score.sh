export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# /home/zhengbaj/tir5/zhiqing/hf_home/hub/models--allenai--tulu-2-dpo-70b/snapshots/0ab5c875f0070d5aee8d36bc55f41de440a13f02
# allenai/tulu-2-dpo-70b

# python -u generate_honesty_score.py \
#     --model_name_or_path "/home/zhengbaj/tir5/zhiqing/hf_home/hub/models--allenai--tulu-2-dpo-70b/snapshots/0ab5c875f0070d5aee8d36bc55f41de440a13f02" \
#     --tokenizer_name "allenai/tulu-2-dpo-70b" \
#     --model_tag "tulu-2" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_scored.jsonl" \
#     --layer_start 30 \
#     --layer_end 50 \
#     --start_answer_token "istant"

# mean: 0.8724035608308606
# min: 0.9276706231454006
# prompt: 0.8683234421364985

# /data/models/huggingface/meta-llama/Llama-2-7b-chat-hf

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --model_tag "llama-2-7b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_scored.jsonl" \
#     --layer_start 10 \
#     --layer_end 20 \
#     --start_answer_token "INST"

# mean: 0.7844955489614244
# min: 0.7752225519287834
# prompt: 0.6973293768545994

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --model_tag "llama-2-13b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_scored.jsonl" \
#     --layer_start 20 \
#     --layer_end 30 \
#     --start_answer_token "INST"

# mean: 0.8635014836795252
# min: 0.9094955489614244
# prompt: 0.7106824925816023

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --model_tag "llama-2-70b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_entity.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio_scored.jsonl" \
#     --layer_start 20 \
#     --layer_end 40 \
#     --layer_step 2 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --start_answer_token "INST"

# mean: 0.8497774480712166
# min: 0.8672106824925816
# prompt: 0.69473293768546
