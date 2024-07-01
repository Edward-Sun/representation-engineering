from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
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
    do_test: bool = False,
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

        for idx in range(1, min(len(true_tokens), len(false_tokens)) - 1):
            true_truncated_tokens = true_tokens[:-idx]
            false_truncated_tokens = false_tokens[:-idx]

            remaining_1 = tuple(true_tokens[-idx:])
            ramaining_2 = tuple(false_tokens[-idx:])

            if remaining_1 != ramaining_2:
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
                                f"{user_tag}{input_str}\n{assistant_tag}"
                                + f"{output_str_1}"
                            )
                        else:
                            honest_statements.append(
                                f"{user_tag}{input_str} {assistant_tag} "
                                + f"{output_str_1}"
                            )
                    else:
                        honest_statements.append(
                            f"{user_tag} {input_str} {assistant_tag} "
                            + f"{output_str_1}"
                        )

                    if user_tag.endswith("\n"):
                        if assistant_tag.endswith("\n"):
                            untruthful_statements.append(
                                f"{user_tag}{f_input_str}\n{assistant_tag}"
                                + f"{f_output_str_1}"
                            )
                        else:
                            untruthful_statements.append(
                                f"{user_tag}{f_input_str} {assistant_tag} "
                                + f"{f_output_str_1}"
                            )
                    else:
                        untruthful_statements.append(
                            f"{user_tag} {f_input_str} {assistant_tag} "
                            + f"{f_output_str_1}"
                        )

    # Create training data
    # ntrain = 512
    combined_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]

    random.shuffle(combined_data)

    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        if do_test:
            train_labels.append([True, False])
        else:
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


def summarize_detection_results(
    input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token=":"
):
    # Define words and their colors
    words = [token.replace("▁", " ") for token in input_ids]

    iter = 0

    selected_concepts = ["honesty"]
    norm_style = ["mean"]

    scores = []

    for rep, n_style in zip(selected_concepts, norm_style):

        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean + 5 * std) | (rep_scores < mean - 5 * std)] = (
            mean  # get rid of outliers
        )

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD  # change this for threshold
        if "flip" in n_style:
            rep_scores = -rep_scores

        rep_scores[np.abs(rep_scores) < 0.0] = 0

        started = False

        for word, score in zip(words[5:], rep_scores[5:]):

            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue

            scores.append(score)

        iter += 1

    return scores


def prepare_dataset(entity_data_path: str):
    bio_data = []

    with open(entity_data_path, "r") as f:
        data = json.load(f)
        random.shuffle(data)
        for item in data:
            entity_pos = item["entity_pos"]
            entity_neg = item["entity_neg"]
            bio_data.append(
                f"Please write a biography paragraph of {entity_pos} with lower case."
                + "\n"
                + item["positive"]
            )
            bio_data.append(
                f"Please write a biography paragraph of {entity_neg} with lower case."
                + "\n"
                + item["negative"]
            )

    entity_df_path = entity_data_path.replace(".json", ".csv")

    df = pd.DataFrame(columns=["statement", "label"])

    for i, sent in enumerate(bio_data):
        if i % 2 == 0:
            df.loc[i] = sent, 1
        else:
            df.loc[i] = sent, 0

    print("Len: DF", len(df))
    # truncate df to 1000
    df = df[:1000]
    print("Len: DF (truncated)", len(df))

    df.to_csv(entity_df_path)
    return entity_df_path


def evaluate(args):
    layers = range(-args.layer_start, -args.layer_end, -args.layer_step)
    rep_reader_scores_dict = {}
    rep_reader_scores_mean_dict = {}

    with open(args.bio_test_data_path, "r") as f:
        test_data = json.load(f)

    def get_score(chosen_str):
        input_ids = tokenizer.tokenize(chosen_str)

        results = []

        ice_pos = slice(-len(input_ids), None)
        H_tests = rep_reading_pipeline(
            [chosen_str],
            rep_reader=honesty_rep_reader,
            rep_token=ice_pos,
            hidden_layers=hidden_layers,
        )

        for ice_pos in range(len(input_ids)):
            ice_pos = -len(input_ids) + ice_pos
            ret_dict = {}
            for layer in H_tests[0]:
                ret_dict[layer] = H_tests[0][layer][:, ice_pos]
            results.append([ret_dict])

        honesty_scores = []
        honesty_scores_means = []
        for pos in range(len(results)):
            tmp_scores = []
            tmp_scores_all = []
            for layer in hidden_layers:
                tmp_scores_all.append(
                    results[pos][0][layer][0]
                    * honesty_rep_reader.direction_signs[layer][0]
                )
                if layer in layers:
                    tmp_scores.append(
                        results[pos][0][layer][0]
                        * honesty_rep_reader.direction_signs[layer][0]
                    )
            honesty_scores.append(tmp_scores_all)
            honesty_scores_means.append(np.mean(tmp_scores))

        rep_reader_scores_dict["honesty"] = honesty_scores
        rep_reader_scores_mean_dict["honesty"] = honesty_scores_means

        THRESHOLD = 0.0  # change this to alter sensitivity
        return summarize_detection_results(
            input_ids,
            rep_reader_scores_mean_dict,
            THRESHOLD,
            start_answer_token=args.start_answer_token,
        )[1:]

    save_path = args.output_score_path.replace(".jsonl", f"_{args.model_tag}.jsonl")

    print("Save Path:", save_path)

    with open(save_path, "w") as f:
        for item in tqdm(test_data):
            entity = item["entity"]
            if user_tag.endswith("\n"):
                if assistant_tag.endswith("\n"):
                    pos_sent = f"{user_tag}Please write a biography paragraph of {entity} with lower case.\n{assistant_tag}{item['positive']}"
                    neg_sent = f"{user_tag}Please write a biography paragraph of {entity} with lower case.\n{assistant_tag}{item['negative']}"
                else:
                    pos_sent = f"{user_tag}Please write a biography paragraph of {entity} with lower case. {assistant_tag} {item['positive']}"
                    neg_sent = f"{user_tag}Please write a biography paragraph of {entity} with lower case. {assistant_tag} {item['negative']}"
            else:
                pos_sent = f"{user_tag} Please write a biography paragraph of {entity} with lower case. {assistant_tag} {item['positive']}"
                neg_sent = f"{user_tag} Please write a biography paragraph of {entity} with lower case. {assistant_tag} {item['negative']}"

            item["positive_score"] = get_score(pos_sent)
            item["negative_score"] = get_score(neg_sent)

            f.write(json.dumps(item) + "\n")
            f.flush()


def evaluate_prompting_baseline(args, pos_id, neg_id):
    with open(args.bio_test_data_path, "r") as f:
        test_data = json.load(f)

    print(tokenizer("Bio A").input_ids, tokenizer("Bio B").input_ids)
    # 365, 330

    acc = []

    for item in tqdm(test_data):
        if user_tag.endswith("\n"):
            if assistant_tag.endswith("\n"):
                sent1 = (
                    f"{user_tag}Please tell me which biography paragraph is more truthful.\n\n"
                    f"Bio A: {item['positive']}\n\nBio B: {item['negative']}\n"
                    f"{assistant_tag}The truthful biography paragraph is Bio"
                )
                sent2 = (
                    f"{user_tag}Please tell me which biography paragraph is more truthful.\n\n"
                    f"Bio A: {item['negative']}\n\nBio B: {item['positive']}\n"
                    f"{assistant_tag}The truthful biography paragraph is Bio"
                )
            else:
                sent1 = (
                    f"{user_tag}Please tell me which biography paragraph is more truthful.\n\n"
                    f"Bio A: {item['positive']}\n\nBio B: {item['negative']} {assistant_tag} The truthful biography paragraph is Bio"
                )
                sent2 = (
                    f"{user_tag}Please tell me which biography paragraph is more truthful.\n\n"
                    f"Bio A: {item['negative']}\n\nBio B: {item['positive']} {assistant_tag} The truthful biography paragraph is Bio"
                )
        else:
            sent1 = (
                f"{user_tag} Please tell me which biography paragraph is more truthful.\n\n"
                f"Bio A: {item['positive']}\n\nBio B: {item['negative']} {assistant_tag} The truthful biography paragraph is Bio"
            )
            sent2 = (
                f"{user_tag} Please tell me which biography paragraph is more truthful.\n\n"
                f"Bio A: {item['negative']}\n\nBio B: {item['positive']} {assistant_tag} The truthful biography paragraph is Bio"
            )

        # get probability of next token
        score_pos = 0.0
        score_neg = 0.0
        token_score = model(
            **tokenizer(sent1, return_tensors="pt").to(model.device)
        ).logits[0, -1]
        score_pos += token_score[pos_id].item()
        score_neg += token_score[neg_id].item()

        token_score = model(
            **tokenizer(sent2, return_tensors="pt").to(model.device)
        ).logits[0, -1]
        score_pos += token_score[neg_id].item()
        score_neg += token_score[pos_id].item()

        if score_pos > score_neg:
            acc.append(1)
        else:
            acc.append(0)

    print(np.mean(acc))


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
        "--model_tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--entity_data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--entity_test_data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--bio_test_data_path", type=str, required=True, help="Bio test data path"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--start_answer_token", type=str, required=True, help="Start answer token"
    )
    parser.add_argument(
        "--layer_start", type=int, default=30, help="Start layer for evaluation"
    )
    parser.add_argument(
        "--layer_end", type=int, default=50, help="End layer for evaluation"
    )
    parser.add_argument(
        "--layer_step", type=int, default=1, help="Step for layer evaluation"
    )
    parser.add_argument(
        "--output_score_path", type=str, required=True, help="Output score path"
    )

    args = parser.parse_args()

    # Load the model and tokenizer

    model_name_or_path = args.model_name_or_path

    if "tulu-2" in model_name_or_path:
        user_tag = "<|user|>\n"
        assistant_tag = "<|assistant|>\n"
    elif "Llama-2" in model_name_or_path and "chat-hf" in model_name_or_path:
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

    print(tokenizer("Bio A").input_ids, tokenizer("Bio B").input_ids)
    pos_id = int(tokenizer("Bio A").input_ids[-1])
    neg_id = int(tokenizer("Bio B").input_ids[-1])

    entity_df_path = prepare_dataset(args.entity_data_path)
    entity_test_df_path = prepare_dataset(args.entity_test_data_path)

    print("Entity DF Path:", entity_df_path)
    print("Entity Test DF Path:", entity_test_df_path)

    entity_dataset = ours_honesty_function_dataset(
        entity_df_path, tokenizer, user_tag, assistant_tag, ntrain=1024
    )
    entity_test_dataset = ours_honesty_function_dataset(
        entity_test_df_path,
        tokenizer,
        user_tag,
        assistant_tag,
        ntrain=512,
        do_test=True,
    )

    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    # n_difference = 1
    # direction_method = 'pca'
    direction_method = "cluster_mean"
    n_difference = 0
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    honesty_rep_reader = rep_reading_pipeline.get_directions(
        entity_dataset["train"]["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference,
        train_labels=entity_dataset["train"]["labels"],
        direction_method=direction_method,
        batch_size=args.train_batch_size,
    )

    H_tests = rep_reading_pipeline(
        entity_test_dataset["train"]["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        rep_reader=honesty_rep_reader,
        batch_size=args.eval_batch_size,
    )

    results = {layer: {} for layer in hidden_layers}
    rep_readers_means = {}
    rep_readers_means["honesty"] = {layer: 0 for layer in hidden_layers}

    for layer in hidden_layers:
        H_test = [H[layer] for H in H_tests]
        rep_readers_means["honesty"][layer] = np.mean(H_test)
        H_test = [H_test[i : i + 2] for i in range(0, len(H_test), 2)]

        sign = honesty_rep_reader.direction_signs[layer]

        eval_func = min if sign == -1 else max
        cors = np.mean([eval_func(H) == H[0] for H in H_test])

        results[layer] = cors

    # plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
    # plt.show()
    # save plot to file

    plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Honesty Detection Accuracy")
    plt.savefig(
        "honesty_detection_accuracy.png".replace(".png", f"_{args.model_tag}.png")
    )

    evaluate(args)
    evaluate_prompting_baseline(args, pos_id, neg_id)
