export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --model_tag "llama-2-7b-new" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 10 \
#     --layer_end 15 \
#     --last_token_only \
#     --start_answer_token "INST"
