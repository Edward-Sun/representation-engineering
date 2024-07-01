export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --model_tag "llama-2-7b-facts-5-15" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/representation-engineering/data/facts/facts_true_false.csv" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 5 \
#     --layer_end 15 \
#     --last_token_only \
#     --start_answer_token "INST"

# 10-15 result: 0.5067319461444308
# 5-15 result: 0.5275397796817626

python -u generate_honesty_score.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
    --model_tag "llama-2-13b-facts-5-15" \
    --entity_data_path "/home/zhengbaj/tir5/zhiqing/representation-engineering/data/facts/facts_true_false.csv" \
    --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
    --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
    --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
    --layer_start 5 \
    --layer_end 15 \
    --last_token_only \
    --start_answer_token "INST"

# 15-20 result: 0.5177478580171359
# 5-15 result:
