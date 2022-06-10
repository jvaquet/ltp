import json

from sklearn.model_selection import train_test_split


def read_data(file_name):
    return json.loads(open(file_name, "r", encoding="utf-8").read())


def split_data(data, is_binary=True):
    labeled_data = data.copy()
    if is_binary:
        for sample in labeled_data:
            if sample["label"] != "claim":
                sample["label"] = "other"

    X, y = (
        [sample["text"] for sample in labeled_data],
        [sample["label"] for sample in labeled_data],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    return X_train, X_test, y_train, y_test


def extract_classes(X_train, y_train):
    claims = [
        {"text": data, "label": label}
        for data, label in zip(X_train, y_train)
        if label == "claim"
    ]
    premises = [
        {"text": data, "label": label}
        for data, label in zip(X_train, y_train)
        if label == "premise"
    ]
    others = [
        {"text": data, "label": label}
        for data, label in zip(X_train, y_train)
        if label == "other"
    ]
    return claims, premises, others


def write_to_json(data, file_name):
    with open(file_name, "w") as f:
        f.write(json.dumps(data, indent=4))


def write_to_pickle(data, file_name):
    import pickle

    with open(file_name, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


prompt_1_binary = """
  Each item in the following list contains text and its argument type. Argument type can be either 1. claim or 2. other.
  Text: {} (Argument Type: {}) 
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {}) 
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {}) 
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: <mask>)
"""

prompt_1_3_label = """
  Each item in the following list contains text and its argument type. Argument type can be either 1. claim, 2. premise, or 3. other.
  Text: {} (Argument Type: {}) 
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {}) 
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: {}) 
  Text: {} (Argument Type: {})
  Text: {} (Argument Type: <mask>)
"""

prompt_2_binary = """
  Each item in the following list contains text and its argument type. Argument type can be either 1. claim or 2. other.
  Text: '{}' Is this a 1. claim or 2. other? This is a {} 
  Text: '{}' Is this a 1. claim or 2. other? This is a {}
  Text: '{}' Is this a 1. claim or 2. other? This is a {}
  Text: '{}' Is this a 1. claim or 2. other? This is a {} 
  Text: '{}' Is this a 1. claim or 2. other? This is a {}
  Text: '{}' Is this a 1. claim or 2. other? This is a {}
  Text: '{}' Is this a 1. claim or 2. other? This is a {}
  Text: '{}' Is this a 1. claim or 2. other? This is a {}
  Text: '{}' Is this a 1. claim or 2. other? This is a <mask>)
"""

prompt_2_3_label = """
  Each item in the following list contains text and its argument type. Argument type can be either 1. claim, 2. premise, or 3. other.
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {} 
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {}
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {}
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {} 
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {}
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {}
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {}
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a {}
  Text: '{}' Is this a 1. claim, 2. premise, or 3. other? This is a <mask>)
"""
