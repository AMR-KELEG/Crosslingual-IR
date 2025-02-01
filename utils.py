import os
import json
import pandas as pd
from pathlib import Path

LANGS = ["ar", "en", "de"]
DATASET_DIR = Path("data", "xquad-r")
os.makedirs(str(DATASET_DIR), exist_ok=True)


def parse_xquard_r(lang):
    with open(str(DATASET_DIR / f"{lang}.json"), "r") as f:
        data = json.load(f)
    questions_paragraphs = {}
    for topic in data:
        for paragraph in topic["paragraphs"]:
            for qa in paragraph["qas"]:
                questions_paragraphs[qa["id"]] = {
                    "title": topic["title"],
                    f"question_{lang}": qa["question"],
                    f"paragraph_{lang}": paragraph["context"],
                }
    return pd.DataFrame(questions_paragraphs).T
