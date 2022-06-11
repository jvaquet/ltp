import argparse

from base_prompt_engg import evaluate_base_prompt_engg, load_pipeline
from baseline import *
from preprocess_claim_identification import (convert_to_json,
                                             create_labeled_data,
                                             get_file_names, pre_process)
from prompt_fine_tuning import *
from utils import *

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--account_id",
        default="nouman-10",
        type=str,
        help="Account Id to save the model on",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = create_arg_parser()
    
    print("Preprocessing data...")
    file_names = get_file_names()
    labeled_data = create_labeled_data(file_names)
    labeled_data = pre_process(labeled_data)
    json_data = convert_to_json(labeled_data)
    write_to_json(json_data, "./claim_identification_data.json")
    write_to_pickle(json_data, "./claim_identification_data.p")
    print("Pre-processing Done!")

    data = read_data("./labeled_data.json")

    print("Doing 3-Label Classification (Claims vs Premises vs Others)")
    X_train, X_test, y_train, y_test = split_data(data, is_binary=False)
    claims, premises, others = extract_classes(X_train, y_train)

    train_and_evaluate_baseline_model(X_train, y_train, X_test, y_test, num_labels=3)

    print("Results with Prompt Engineering (no fine-tuning)")
    print("Prompt 1:")
    evaluate_base_prompt_engg(
        "roberta-base",
        X_test,
        y_test,
        claims,
        premises,
        others,
        is_binary=False,
        prompt=prompt_1_3_label,
    )

    print("Prompt 2:")
    evaluate_base_prompt_engg(
        "roberta-base",
        X_test,
        y_test,
        claims,
        premises,
        others,
        is_binary=False,
        prompt=prompt_2_3_label,
    )

    print("Doing Binary Classification (Claims vs Others)")
    X_train, X_test, y_train, y_test = split_data(data, is_binary=True)
    claims, premises, others = extract_classes(X_train, y_train)

    train_and_evaluate_baseline_model(X_train, y_train, X_test, y_test, num_labels=2)

    print("Results with Prompt Engineering (no fine-tuning)")
    print("Prompt 1:")
    evaluate_base_prompt_engg(
        "roberta-base",
        X_test,
        y_test,
        claims,
        premises,
        others,
        is_binary=True,
        prompt=prompt_1_binary,
    )

    print("Prompt 2:")
    evaluate_base_prompt_engg(
        "roberta-base",
        X_test,
        y_test,
        claims,
        premises,
        others,
        is_binary=True,
        prompt=prompt_2_binary,
    )

    print("Results with Prompt Engineering (fine-tuned)")
    print("Only Binary Classification with Prompt 2:")
    model, tokenizer = load_model("roberta-base"), load_tokenizer("roberta-base")
    train_dataset, test_dataset = (
        generate_dataset(X_train, y_train),
        generate_dataset(X_test, y_test),
    )

    dataset = combine_train_test(train_dataset, test_dataset)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    lm_datasets = tokenized_dataset.map(group_texts, batched=True)

    tf_train_dataset, tf_test_dataset = create_tf_datasets(lm_datasets, tokenizer)

    model, callbacks = compile_model(model, tokenizer, len(tf_train_dataset))

    print("Perplexity before training:")
    print(calculate_perplexity(model, tf_test_dataset))

    print("Training the model...")
    model = train_model(model, callbacks, tf_train_dataset)

    print("Perplexity after training:")
    print(calculate_perplexity(model, tf_test_dataset))

    print("Evaluating the fine-tuned model...")
    model = load_pipeline(f"{args.account_id}/robertabase-claims-2")
    evaluate_trained_model(model, X_test, y_test)
