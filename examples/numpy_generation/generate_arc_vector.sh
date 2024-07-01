export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_arc_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc.h5"

# python -u generate_arc_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc.h5" \
#     --eval_batch_size 32

# python -u generate_arc_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_arc.h5"

# python -u generate_arc_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc_v2.h5" \
#     --prompt_style "v2"

python -u generate_arc_vector.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
    --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc_v2.h5" \
    --prompt_style "v2" \
    --eval_batch_size 32
