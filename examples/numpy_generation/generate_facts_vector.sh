export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_facts_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/representation-engineering/data/facts/facts_true_false.csv" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts.h5"

# python -u generate_facts_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/representation-engineering/data/facts/facts_true_false.csv" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts_last.h5" \
#     --last_token_only

# python -u generate_facts_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/representation-engineering/data/facts/facts_true_false.csv" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts_v2_last.h5" \
#     --prompt_style "v2" \
#     --last_token_only

python -u generate_facts_vector.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
    --data_path "/home/zhengbaj/tir5/zhiqing/representation-engineering/data/facts/facts_true_false.csv" \
    --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_facts_v2_last.h5" \
    --prompt_style "v2" \
    --last_token_only \
    --eval_batch_size 32