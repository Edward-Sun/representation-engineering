export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

python -u generate_honesty_vector.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
    --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_wiki_bio/test_bio.json" \
    --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_bio_test_truthful.h5" \
    --last_token_only \
    --bio_data \
    --eval_batch_size 32
