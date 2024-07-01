export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --model_tag "llama-2-7b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 10 \
#     --layer_end 15 \
#     --start_answer_token "INST"

# mean: 0.49326805385556916
# min: 0.5581395348837209
# prompt: 0.6817625458996328
# Method: mean, Accuracy: 0.5165238678090576
# Method: min, Accuracy: 0.5581395348837209

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --model_tag "llama-2-7b-hard" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 10 \
#     --layer_end 15 \
#     --start_answer_token "INST"

# mean: 0.5018359853121175
# min: 0.5373317013463892
# prompt: 0.6817625458996328
# Method: mean, Accuracy: 0.5165238678090576
# Method: min, Accuracy: 0.5312117503059975

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --model_tag "llama-2-13b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 15 \
#     --layer_end 20 \
#     --start_answer_token "INST"

# mean: 0.5361077111383109
# min: 0.6034271725826194
# prompt: 0.6548347613219094
# Method: mean, Accuracy: 0.591187270501836
# Method: min, Accuracy: 0.6132190942472461

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --model_tag "llama-2-13b-hard" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 15 \
#     --layer_end 20 \
#     --start_answer_token "INST"

# hard
# mean: 0.5752753977968176
# min: 0.6070991432068543
# prompt: 0.6548347613219094
# Method: mean, Accuracy: 0.6034271725826194
# Method: min, Accuracy: 0.609547123623011

# python -u generate_honesty_score.py \
#     --model_name_or_path "/home/zhengbaj/tir5/zhiqing/hf_home/hub/models--allenai--tulu-2-dpo-70b/snapshots/0ab5c875f0070d5aee8d36bc55f41de440a13f02" \
#     --tokenizer_name "allenai/tulu-2-dpo-70b" \
#     --model_tag "tulu-2" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 20 \
#     --layer_end 40 \
#     --layer_step 2 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --start_answer_token "istant"

# mean: 0.44430844553243576
# min: 0.3818849449204406
# prompt: 0.6682986536107711

# python -u generate_honesty_score.py \
#     --model_name_or_path "/home/zhengbaj/tir5/zhiqing/hf_home/hub/models--allenai--tulu-2-dpo-70b/snapshots/0ab5c875f0070d5aee8d36bc55f41de440a13f02" \
#     --tokenizer_name "allenai/tulu-2-dpo-70b" \
#     --model_tag "tulu-2-hard" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 20 \
#     --layer_end 40 \
#     --layer_step 2 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --start_answer_token "istant"

# mean: 0.32802937576499386
# min: 0.2913096695226438
# prompt: 0.6682986536107711

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --model_tag "llama-2-70b" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 20 \
#     --layer_end 40 \
#     --layer_step 2 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --start_answer_token "INST"

# # mean: 0.5263157894736842
# # min: 0.6217870257037944
# # prompt: 0.6768665850673194
# Method: mean, Accuracy: 0.6070991432068543
# Method: min, Accuracy: 0.631578947368421

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --model_tag "llama-2-70b-hard" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_hard.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 20 \
#     --layer_end 40 \
#     --layer_step 2 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --start_answer_token "INST"

# mean: 0.5055079559363526
# min: 0.5960832313341493
# prompt: 0.6768665850673194
# Method: mean, Accuracy: 0.5752753977968176
# Method: min, Accuracy: 0.6083231334149327

python -u generate_honesty_score.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-hf" \
    --tokenizer_name "meta-llama/Llama-2-70b-hf" \
    --model_tag "llama-2-70b-non-chat" \
    --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
    --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
    --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
    --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
    --layer_start 20 \
    --layer_end 40 \
    --layer_step 2 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --start_answer_token "INST"

# mean:
# min:
# prompt:
