export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_medqa_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_medqa/medqa_test.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_medqa_v3.h5" \
#     --prompt_style "v3" \
#     --eval_batch_size 64

# python -u generate_medqa_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_medqa/medqa_test.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_medqa_v4.h5" \
#     --prompt_style "v4" \
#     --eval_batch_size 64

python -u generate_medqa_vector.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
    --data_path "/home/zhengbaj/tir5/zhiqing/data/dataset_medqa/medqa_test.json" \
    --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_medqa_v3.h5" \
    --prompt_style "v3" \
    --eval_batch_size 32