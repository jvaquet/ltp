import glob
import json

from bs4 import BeautifulSoup as Bs


def get_arguments(response):
    all_text = response.text.replace("\n", "").strip()

    claim_texts = [claim.text.strip() for claim in response.find_all("claim")]
    premise_texts = [premise.text.strip() for premise in response.find_all("premise")]

    premise_texts = [
        premise_text
        for premise_text in premise_texts
        if [
            text
            for text in claim_texts
            if text in premise_text and text != premise_text
        ]
        == []
    ]
    premise_texts = [
        premise_text
        for premise_text in premise_texts
        if [
            text
            for text in premise_texts
            if text in premise_text and text != premise_text
        ]
        == []
    ]

    claim_texts = [
        claim_text
        for claim_text in claim_texts
        if [text for text in claim_texts if text in claim_text and text != claim_text]
        == []
    ]
    claim_texts = [
        claim_text
        for claim_text in claim_texts
        if [text for text in premise_texts if text in claim_text and text != claim_text]
        == []
    ]

    return all_text, claim_texts, premise_texts


def get_responses(file_path, file_num, delta_type):
    soup = Bs(
        open(
            file_path.format(delta_type, file_num, "xml"), "r", encoding="utf-8"
        ).read(),
        "lxml",
    )

    responses = [soup.find("op")] + soup.find_all("reply")
    return responses


def get_file_names():
    negative_file_nums = [
        (file_name.split("/")[-1].split(".")[0], "negative")
        for file_name in glob.glob("./change-my-view-modes/v2.0/negative/*.xml")
    ]
    positive_file_nums = [
        (file_name.split("/")[-1].split(".")[0], "positive")
        for file_name in glob.glob("./change-my-view-modes/v2.0/positive/*.xml")
    ]

    return negative_file_nums + positive_file_nums


def create_labeled_data(file_names):
    file_path = "./change-my-view-modes/v2.0/{}/{}.{}"
    labeled_data = []

    file_names = get_file_names()
    for file_num, delta_type in file_names:
        responses = get_responses(file_path, file_num, delta_type)
        for response in responses:
            all_text, claim_texts, premise_texts = get_arguments(response)
            while claim_texts or premise_texts:

                current_index = 0
                claim_start = len(all_text) + 1
                premise_start = len(all_text) + 1
                if claim_texts:
                    claim_start = all_text.find(claim_texts[0])

                if premise_texts:
                    premise_start = all_text.find(premise_texts[0])

                if claim_start and premise_start:
                    current_index = (
                        premise_start if premise_start < claim_start else claim_start
                    )
                    if len(all_text[:current_index].strip().split()) > 1:
                        labeled_data.append((all_text[:current_index], "other"))

                else:
                    is_premise_first = premise_start < claim_start

                    text = (
                        premise_texts.pop(0) if is_premise_first else claim_texts.pop(0)
                    )
                    label = "premise" if is_premise_first else "claim"

                    labeled_data.append((text, label))

                    current_index = len(text)

                all_text = all_text[current_index:]

    return labeled_data


def pre_process(labeled_data):
    processed_data = list(set(labeled_data))
    processed_data = [data for data in processed_data if len(data[0].split()) > 4]

    return processed_data


def convert_to_json(data):
    json_data = []
    for text, label in data:
        json_data.append({"text": text, "label": label})

    return json_data
