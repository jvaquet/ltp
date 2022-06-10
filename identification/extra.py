def extract_title_claim(soup):
    title_claim = soup.find("title").find("claim")
    return {
        "claim": title_claim.text,
        "type": title_claim.get("type"),
    }


def extract_claims_or_premises(arguments, claim=False, multiple=True):
    argument_type = "claim" if claim else "premise"
    if not multiple:
        return {
            argument.get("id"): {
                argument_type: argument.text,
                "type": argument.get("type"),
                "id": argument.get("id"),
                "rel": argument.get("rel"),
                "ref": argument.get("ref"),
            }
            for argument in arguments
        }
    else:
        return {
            argument.get("id"): {
                argument_type: argument.text,
                "type": argument.get("type"),
                "rel": argument.get("rel"),
                "ref": argument.get("ref"),
            }
            for argument in arguments
            for argument in argument.find_all(argument_type)
        }


import glob

negative_files = glob.glob("change-my-view-modes/v2.0/negative/*.xml")
positive_files = glob.glob("change-my-view-modes/v2.0/positive/*.xml")
negative_records = []
claims = []
for file_name in negative_files:
    soup = Bs(open(file_name, "r", encoding="utf-8").read(), "lxml")
    record = {
        "id": file_name.split("/")[-1].split(".")[0],
        "title": extract_title_claim(soup),
        "op_claims": extract_claims_or_premises(
            soup.find("op").find_all("claim"), claim=True, multiple=False
        ),
        "op_premises": extract_claims_or_premises(
            soup.find("op").find_all("premise"), multiple=False
        ),
        "claims": extract_claims_or_premises(soup.find_all("reply"), claim=True),
        "premises": extract_claims_or_premises(soup.find_all("reply")),
    }
    claims.extend([record["claims"][key] for key in record["claims"].keys()])
    claims.extend([record["op_claims"][key] for key in record["op_claims"].keys()])
    negative_records.append(record)

positive_records = []
for file_name in positive_files:
    soup = Bs(open(file_name, "r", encoding="utf-8").read(), "lxml")
    record = {
        "id": file_name.split("/")[-1].split(".")[0],
        "title": extract_title_claim(soup),
        "op_claims": extract_claims_or_premises(
            soup.find("op").find_all("claim"), claim=True, multiple=False
        ),
        "op_premises": extract_claims_or_premises(
            soup.find("op").find_all("premise"), claim=True, multiple=False
        ),
        "claims": extract_claims_or_premises(soup.find_all("reply"), claim=True),
        "premises": extract_claims_or_premises(soup.find_all("reply")),
    }

    claims.extend([record["claims"][key] for key in record["claims"].keys()])
    claims.extend([record["op_claims"][key] for key in record["op_claims"].keys()])
    positive_records.append(record)
