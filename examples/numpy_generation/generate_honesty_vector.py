from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import h5py
import numpy as np

import json
import random
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


def ours_honesty_function_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    user_tag: str = "",
    assistant_tag: str = "",
    seed: int = 0,
    ntrain: int = 512,
    last_token_only: bool = False,
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

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1]["statement"].values.tolist()
    false_statements = df[df["label"] == 0]["statement"].values.tolist()

    honest_statements = []
    untruthful_statements = []

    # Process statements
    for true_statement, false_statement in tqdm(zip(true_statements, false_statements)):
        true_tokens = tokenizer.tokenize(true_statement)
        false_tokens = tokenizer.tokenize(false_statement)

        for_limit = min(len(true_tokens), len(false_tokens)) - 1
        for_limit = min(for_limit, 5)
        if last_token_only:
            for_limit = 2

        for idx in range(1, for_limit):
            true_truncated_tokens = true_tokens[:-idx]
            false_truncated_tokens = false_tokens[:-idx]

            remaining_1 = tuple(true_tokens[-idx:])
            ramaining_2 = tuple(false_tokens[-idx:])

            if (
                remaining_1 != ramaining_2
                or true_truncated_tokens[-1] != false_truncated_tokens[-1]
            ):
                continue

            if len(true_truncated_tokens) > 128 or len(false_truncated_tokens) > 128:
                continue

            true_truncated_statement = tokenizer.convert_tokens_to_string(
                true_truncated_tokens
            )
            false_truncated_statement = tokenizer.convert_tokens_to_string(
                false_truncated_tokens
            )

            if (
                len(true_truncated_statement.split("\n")) == 2
                and len(false_truncated_statement.split("\n")) == 2
            ):
                # if true_truncated_statement.split("\n")[-1] == false_truncated_statement.split("\n")[-1]:
                input_str, output_str_1 = true_truncated_statement.split("\n", 1)
                f_input_str, f_output_str_1 = false_truncated_statement.split("\n", 1)
                if output_str_1 != f_output_str_1:

                    if user_tag.endswith("\n"):
                        if assistant_tag.endswith("\n"):
                            honest_statements.append(
                                f"{user_tag}{input_str}\n{assistant_tag}{output_str_1}"
                            )
                        else:
                            honest_statements.append(
                                f"{user_tag}{input_str} {assistant_tag} {output_str_1}"
                            )
                    else:
                        honest_statements.append(
                            f"{user_tag} {input_str} {assistant_tag} {output_str_1}"
                        )

                    if user_tag.endswith("\n"):
                        if assistant_tag.endswith("\n"):
                            untruthful_statements.append(
                                f"{user_tag}{f_input_str}\n{assistant_tag}{f_output_str_1}"
                            )
                        else:
                            untruthful_statements.append(
                                f"{user_tag}{f_input_str} {assistant_tag} {f_output_str_1}"
                            )
                    else:
                        untruthful_statements.append(
                            f"{user_tag} {f_input_str} {assistant_tag} {f_output_str_1}"
                        )

    # Create training data
    combined_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]

    random.shuffle(combined_data)

    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        if d[0] == true_s:
            train_labels.append([True, False])
        else:
            train_labels.append([False, True])

    train_data = np.concatenate(train_data).tolist()

    print(f"Train data: {len(train_data)}")

    return {
        "train": {"data": train_data, "labels": train_labels},
    }


def ours_test_function_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    user_tag: str = "",
    assistant_tag: str = "",
    seed: int = 0,
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

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1]["statement"].values.tolist()
    false_statements = df[df["label"] == 0]["statement"].values.tolist()

    assert len(true_statements) > 0
    assert len(false_statements) > 0

    honest_statements = []
    untruthful_statements = []

    # Process statements
    for true_statement, false_statement in tqdm(zip(true_statements, false_statements)):
        true_tokens = tokenizer.tokenize(true_statement)
        false_tokens = tokenizer.tokenize(false_statement)
        idx = 1

        true_truncated_tokens = true_tokens[:-idx]
        false_truncated_tokens = false_tokens[:-idx]

        true_truncated_statement = tokenizer.convert_tokens_to_string(
            true_truncated_tokens
        )
        false_truncated_statement = tokenizer.convert_tokens_to_string(
            false_truncated_tokens
        )

        assert (
            len(true_truncated_statement.split("\n")) == 2
            and len(false_truncated_statement.split("\n")) == 2
        )

        input_str, output_str_1 = true_truncated_statement.split("\n", 1)
        f_input_str, f_output_str_1 = false_truncated_statement.split("\n", 1)
        if user_tag.endswith("\n"):
            if assistant_tag.endswith("\n"):
                honest_statements.append(
                    f"{user_tag}{input_str}\n{assistant_tag}{output_str_1}"
                )
            else:
                honest_statements.append(
                    f"{user_tag}{input_str} {assistant_tag} {output_str_1}"
                )
        else:
            honest_statements.append(
                f"{user_tag} {input_str} {assistant_tag} {output_str_1}"
            )

        if user_tag.endswith("\n"):
            if assistant_tag.endswith("\n"):
                untruthful_statements.append(
                    f"{user_tag}{f_input_str}\n{assistant_tag}{f_output_str_1}"
                )
            else:
                untruthful_statements.append(
                    f"{user_tag}{f_input_str} {assistant_tag} {f_output_str_1}"
                )
        else:
            untruthful_statements.append(
                f"{user_tag} {f_input_str} {assistant_tag} {f_output_str_1}"
            )

    # Create training data
    train_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]

    train_labels = []
    for d in train_data:
        train_labels.append([True, False])

    train_data = np.concatenate(train_data).tolist()

    print(f"Train data: {len(train_data)}")

    return {
        "train": {"data": train_data, "labels": train_labels},
    }


def prepare_entity_dataset(
    entity_data_path: str, last_token_only: bool = False, debug: bool = False
):
    bio_data = []

    with open(entity_data_path, "r") as f:
        data = json.load(f)
        random.shuffle(data)
        for item in data:
            bio_data.append(
                f"Consider the correctness of the following biography of an individual: "
                + item["positive"]
                + "\nThe probability the answer being correct is "
            )
            bio_data.append(
                f"Consider the correctness of the following biography of an individual: "
                + item["negative"]
                + "\nThe probability the answer being correct is "
            )

    entity_df_path = entity_data_path.replace(".json", ".csv")

    df = pd.DataFrame(columns=["statement", "label"])

    for i, sent in enumerate(bio_data):
        if i % 2 == 0:
            df.loc[i] = sent, 1
        else:
            df.loc[i] = sent, 0

    print("Len: DF", len(df))
    if debug:
        df = df[:100]
    else:
        if last_token_only:
            df = df[:8000]
        else:
            df = df[:4000]
    print("Len: DF (truncated)", len(df))

    df.to_csv(entity_df_path)
    return entity_df_path


def prepare_bio_dataset(bio_data_path: str):
    bio_data = []

    with open(bio_data_path, "r") as f:
        data = json.load(f)
        random.shuffle(data)
        for item in data:
            bio_data.append(
                f"Consider the correctness of the following biography of an individual: "
                + item["positive"]
                + "\nThe probability the answer being correct is "
            )
            bio_data.append(
                f"Consider the correctness of the following biography of an individual: "
                + item["negative"]
                + "\nThe probability the answer being correct is "
            )

    bio_df_path = bio_data_path.replace(".json", ".csv")

    df = pd.DataFrame(columns=["statement", "label"])

    for i, sent in enumerate(bio_data):
        if i % 2 == 0:
            df.loc[i] = sent, 1
        else:
            df.loc[i] = sent, 0

    df.to_csv(bio_df_path)
    return bio_df_path


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
    parser.add_argument("--bio_data", action="store_true", help="Use bio data")
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

    if args.bio_data:
        df_path = prepare_bio_dataset(args.data_path)
    else:
        df_path = prepare_entity_dataset(
            args.data_path, args.last_token_only, args.debug
        )

    print("DF Path:", df_path)

    if args.bio_data:
        dataset = ours_test_function_dataset(
            df_path,
            tokenizer,
            user_tag,
            assistant_tag,
        )
    else:
        dataset = ours_honesty_function_dataset(
            df_path,
            tokenizer,
            user_tag,
            assistant_tag,
            ntrain=10**9,
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

    # Save the labels
    if args.bio_data:
        print(f"Shape: {save_arr.shape}")
    else:
        labels = dataset["train"]["labels"]
        labels = np.array(labels)

        np.save(args.output_embed_path.replace(".h5", "_labels.npy"), labels)
        print(f"Labels saved at {args.output_embed_path.replace('.h5', '_labels.npy')}")
        print(f"Shape: {save_arr.shape}, {labels.shape}")
