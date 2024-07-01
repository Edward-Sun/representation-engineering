export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --model_tag "llama-2-70b-10-30" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 10 \
#     --layer_end 30 \
#     --layer_step 2 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --start_answer_token "INST"

# python -u generate_honesty_score.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --model_tag "llama-2-70b-30-50" \
#     --entity_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity.json" \
#     --entity_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --bio_test_data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test.json" \
#     --output_score_path "/home/zhengbaj/tir5/zhiqing/data/dataset_truthfulqa/truthfulqa_test_scored.jsonl" \
#     --layer_start 30 \
#     --layer_end 50 \
#     --layer_step 2 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --start_answer_token "INST"
