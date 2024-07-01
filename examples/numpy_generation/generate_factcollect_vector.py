from typing import List, Tuple
import torch
import h5py
import numpy as np

import json

import argparse

from transformers import (
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM,
)
from repe import repe_pipeline_registry

repe_pipeline_registry()


ptrain, ntrain = 500, 500


def ours_facts_function_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    user_tag: str = "",
    assistant_tag: str = "",
    mode: str = "train",
    max_length: int = 256,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.

    Returns:
    - Tuple containing train and test data.
    """
    # Load the data
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # random shuffle
    np.random.seed(seed)
    np.random.shuffle(data)
    new_data = []

    num_pos = 0
    num_neg = 0

    assert mode in ["train", "test"]

    for example in data:
        count_in = False
        if example["label"] == "CORRECT":
            num_pos += 1
            if num_pos <= ptrain:
                count_in = True
        else:
            num_neg += 1
            if num_neg <= ntrain:
                count_in = True

        if not count_in and mode == "train":
            continue

        new_data.append(
            {
                "question": example["article"],
                "choice": example["summary"],
                "label": 1 if example["label"] == "CORRECT" else 0,
            }
        )

    statements = []
    labels = []

    for example in new_data:
        question = example["question"]

        # truncate the question
        tokenized_question = tokenizer.tokenize(question)
        if len(tokenized_question) > max_length:
            tokenized_question = tokenized_question[:max_length]
            question = tokenizer.convert_tokens_to_string(tokenized_question)
            question = question.rsplit(" ", 1)[0] + "..."

        choice = example["choice"]

        question = f"Consider the correctness of the summary to the following article:\nArticle: {question}\nSummary: {choice}."
        truncated_choice = "The probability the summary being accurate is"

        if user_tag.endswith("\n"):
            if assistant_tag.endswith("\n"):
                statements.append(
                    f"{user_tag}{question}\n{assistant_tag}{truncated_choice}"
                )
            else:
                statements.append(
                    f"{user_tag}{question} {assistant_tag} {truncated_choice}"
                )
        else:
            statements.append(
                f"{user_tag} {question} {assistant_tag} {truncated_choice}"
            )

        labels.append(example["label"])

    print(f"Train data: {len(statements)}")

    return {
        "train": {"data": statements, "labels": labels},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate honesty scores")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_embed_path", type=str, required=True, help="Output embedding path"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Max length of the input"
    )
    parser.add_argument("--mode", type=str, default="train", help="Mode")
    parser.add_argument("--debug", action="store_true", help="Use last token only")

    args = parser.parse_args()

    # Load the model and tokenizer

    model_name_or_path = args.model_name_or_path

    if "tulu-2" in model_name_or_path:
        user_tag = "<|user|>\n"
        assistant_tag = "<|assistant|>\n"
    elif "Llama-2" in model_name_or_path and "-hf" in model_name_or_path:
        user_tag = (
            "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\n"
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
        )
        assistant_tag = "[/INST]"
    else:
        raise ValueError("Model not supported")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, device_map="auto"
    )
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures

    tokenizer_name = args.tokenizer_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=use_fast_tokenizer, padding_side="left", legacy=False
    )
    tokenizer.pad_token_id = 0

    dataset = ours_facts_function_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        mode=args.mode,
        max_length=args.max_length,
    )

    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    print(len(dataset["train"]["data"]))
    print(len(dataset["train"]["labels"]))

    H_tests = rep_reading_pipeline(
        dataset["train"]["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        batch_size=args.eval_batch_size,
    )

    # N: number of samples
    # print(len(H_tests[0]))
    # L: number of layers
    # print(H_tests[0][-1].shape)
    # -1 to -L
    # [1, D]: D is the dimension of the representation

    # Save the embeddings
    # [L, N, D]

    N, L, D = len(H_tests), len(H_tests[0]), H_tests[0][-1].shape[1]

    save_arr = np.zeros((L, N, D))

    for i in range(N):
        for j in range(L):
            save_arr[j][i] = H_tests[i][j - L].reshape(-1)

    with h5py.File(args.output_embed_path, "w") as hf:
        hf.create_dataset("embeddings", data=save_arr)

    print(f"Embeddings saved at {args.output_embed_path}")

    labels = dataset["train"]["labels"]
    labels = np.array(labels)

    np.save(args.output_embed_path.replace(".h5", "_labels.npy"), labels)
    print(f"Labels saved at {args.output_embed_path.replace('.h5', '_labels.npy')}")
    print(f"Shape: {save_arr.shape}, {labels.shape}")
