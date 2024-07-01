from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import h5py
import numpy as np

import json
import pandas as pd

import argparse

from transformers import (
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from repe import repe_pipeline_registry

repe_pipeline_registry()


question_rephrases = [
    "Can you provide me with a fact about the world?",
    "Share a fact about the world with me.",
    "Give me a fact about the world.",
    "What is one statement you can tell me about the world?",
    "What is a fact about the world?",
]


def ours_facts_function_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    user_tag: str = "",
    assistant_tag: str = "",
    last_token_only: bool = False,
    prompt_style: str = "v1",
) -> Tuple[List[str], List[str]]:
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """
    # Load the data
    with open(data_path, "r") as f:
        data = pd.read_csv(f)

    # data to json
    data = data.to_dict(orient="records")

    statements = []
    all_labels = []

    for example in data:
        tokens = tokenizer.tokenize(example["statement"])

        if prompt_style == "v1":
            for_limit = len(tokens) - 3
            for_limit = min(for_limit, 5)
            if last_token_only:
                for_limit = 2

            for idx in range(1, for_limit):
                truncated_tokens = tokens[:-idx]
                truncated_choice = tokenizer.convert_tokens_to_string(truncated_tokens)

                for question in question_rephrases:
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
                    all_labels.append(example["label"] == 1)
        elif prompt_style == "v2":
            statement = example["statement"]
            question = (
                f"Answer: {statement} (true or false)\n\n"
                f"I'm in an exam and the above is a true/false question. "
                "I'm not sure whether the answer is true or false. Can you help me?"
            )
            truncated_choice = "Sure! The given answer is"

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
            all_labels.append(example["label"] == 1)
        else:
            raise ValueError("Prompt style not supported")

    print(f"Train data: {len(statements)}")

    return {
        "train": {"data": statements, "labels": all_labels},
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
        "--last_token_only", action="store_true", help="Use last token only"
    )
    parser.add_argument("--prompt_style", type=str, default="v1", help="Prompt style")
    parser.add_argument("--debug", action="store_true", help="Use last token only")

    args = parser.parse_args()

    # Load the model and tokenizer

    model_name_or_path = args.model_name_or_path

    if "tulu-2" in model_name_or_path:
        user_tag = "<|user|>\n"
        assistant_tag = "<|assistant|>\n"
    elif "Llama-2" in model_name_or_path and "-hf" in model_name_or_path:
        user_tag = (
            "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
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
        args.data_path,
        tokenizer,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        prompt_style=args.prompt_style,
        last_token_only=args.last_token_only,
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
