export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_honesty_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_plus.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus.h5"

# python -u generate_honesty_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_plus.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_last.h5" \
#     --last_token_only

# python -u generate_honesty_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_bio_test_truthful.h5" \
#     --last_token_only \
#     --bio_data

python -u generate_honesty_vector.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
    --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio.json" \
    --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_bio_test_truthful.h5" \
    --last_token_only \
    --bio_data

# python -u generate_honesty_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/train_entity_plus.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_entity_plus_last.h5" \
#     --last_token_only \
#     --eval_batch_size 64
