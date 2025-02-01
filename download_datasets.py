import json
import requests
import pandas as pd

from utils import LANGS, DATASET_DIR


def acquire_xquad_r(lang):
    file_url = (
        "https://raw.githubusercontent.com/google-research-datasets"
        + f"/lareqa/refs/heads/master/xquad-r/{lang}.json"
    )
    r = requests.get(file_url)
    with open(str(DATASET_DIR / f"{lang}.json"), "w") as f:
        json.dump(r.json()["data"], f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    for lang in LANGS:
        acquire_xquad_r(lang)
