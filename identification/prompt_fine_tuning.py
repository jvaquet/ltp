import math

import pandas as pd
import tensorflow as tf
from datasets import DatasetDict, load_dataset
from sklearn.metrics import classification_report
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          TFAutoModelForMaskedLM, create_optimizer)
from transformers.keras_callbacks import PushToHubCallback


def load_model(lm_name):
    model = TFAutoModelForMaskedLM.from_pretrained(lm_name)
    model(model.dummy_inputs)

    return model


def load_tokenizer(lm_name):
    return AutoTokenizer.from_pretrained(lm_name)


def generate_dataset(X, y):
    dataset = [
        {
            "text": f"Text: {text}. Is this a 1. claim or 2. other? <mask>",
            "labels": f"Text: {text}. Is this a 1. claim or 2. other? {label}",
        }
        for text, label in zip(X, y)
    ]

    df = pd.DataFrame.from_records(dataset)
    df.to_json("dataset.json", orient="records", lines=True)

    hf_dataset = load_dataset("json", data_files="dataset.json", split="train")

    return hf_dataset


def combine_train_test(train_dataset, test_dataset):
    return DatasetDict({"train": train_dataset, "test": test_dataset})


def tokenize(tokenizer, examples):
    result = tokenizer(examples["text"])
    result["labels"] = tokenizer(examples["labels"])["input_ids"]

    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]

    return result


def group_texts(examples, chunk_size=128):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }

    return result


def create_tf_datasets(datasets, tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, return_tensors="tf", mlm=False
    )

    tf_train_dataset = datasets["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=16,
    )

    tf_eval_dataset = datasets["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=16,
    )

    return tf_train_dataset, tf_eval_dataset


def compile_model(model, tokenizer, num_train_steps):
    optimizer, _ = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=1_000,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model.compile(optimizer=optimizer)

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    callback_1 = PushToHubCallback(
        output_dir=f"robertabase-claims-2", tokenizer=tokenizer
    )
    callback_2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

    return model, [callback_1, callback_2]


def calculate_perplexity(model, tf_eval_dataset):
    eval_loss = model.evaluate(tf_eval_dataset)
    print(f"Perplexity: {math.exp(eval_loss):.2f}")


def train_model(model, callbacks, tf_train_dataset):
    model.fit(tf_train_dataset, epochs=5, callbacks=[callbacks], validation_split=0.1)
    return model


def evaluate_trained_model(model, X_test, y_test):
    prompt = "Text: {} Is this a 1. claim or 2. other?: <mask>"

    y_pred = []
    for text in X_test:
        pred = model(prompt.format(text))[0]["token_str"].strip()
        y_pred.append(pred)

    print(classification_report(y_test, y_pred))
