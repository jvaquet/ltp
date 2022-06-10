import random

from sklearn.metrics import classification_report
from transformers import pipeline


def load_pipeline(lm_name):
    return pipeline("fill-mask", model=lm_name)


def generate_prompt(claims, premises, others, test_text, is_binary, prompt):
    if is_binary:
        selected_args = random.sample(claims, 4) + random.sample(premises, 4)
    else:
        selected_args = (
            random.sample(claims, 3)
            + random.sample(premises, 3)
            + random.sample(others, 2)
        )

    random.shuffle(selected_args)

    format_list = []
    for args in selected_args:
        format_list.append(args["text"])
        format_list.append(args["label"])

    format_list.append(test_text)

    return prompt.format(*format_list)


def evaluate_base_prompt_engg(
    lm_name, X_test, y_test, claims, premises, others, is_binary, prompt
):
    model = load_pipeline(lm_name)

    y_pred = []
    for text in X_test:
        prompt = generate_prompt(claims, premises, others, text, is_binary, prompt)

        pred = model(prompt)[0]["token_str"].strip()
        y_pred.append(pred)

    print(classification_report(y_test, y_pred))
