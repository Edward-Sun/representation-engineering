export HF_HOME="/home/zhengbaj/tir5/zhiqing/hf_home"

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/fact/fact_train_unfiltered.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_factcollect.h5" \
#     --mode "train" \
#     --max_length 512 \
#     --eval_batch_size 64

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/fact/fact_test_unfiltered.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_factcollect_test.h5" \
#     --mode "test" \
#     --max_length 512

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/covidfact/covidfact_test.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_covidfact_test.h5" \
#     --mode "test" \
#     --max_length 512 \
#     --eval_batch_size 64

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/fact/fact_train_unfiltered.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_factcollect.h5" \
#     --mode "train" \
#     --max_length 512 \
#     --eval_batch_size 32

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/fact/fact_test_unfiltered.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_factcollect_test.h5" \
#     --mode "test" \
#     --eval_batch_size 32 \
#     --max_length 512

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-13b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/covidfact/covidfact_test.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_covidfact_test.h5" \
#     --mode "test" \
#     --max_length 512 \
#     --eval_batch_size 32

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/fact/fact_train_unfiltered.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_factcollect.h5" \
#     --mode "train" \
#     --max_length 512 \
#     --eval_batch_size 16

# python -u generate_factcollect_vector.py \
#     --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
#     --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
#     --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/fact/fact_test_unfiltered.json" \
#     --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_factcollect_test.h5" \
#     --mode "test" \
#     --eval_batch_size 16 \
#     --max_length 512

python -u generate_factcollect_vector.py \
    --model_name_or_path "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf" \
    --tokenizer_name "meta-llama/Llama-2-70b-chat-hf" \
    --data_path "/home/zhengbaj/tir5/zhiqing/factkb/data/healthver/healthver_test.json" \
    --output_embed_path "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_healthver_test.h5" \
    --mode "test" \
    --max_length 512 \
    --eval_batch_size 16