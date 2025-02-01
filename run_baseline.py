# Multilingual setup:
## Questions in Arabic
## Paragraphs in any of the languages

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from random import randint
from utils import parse_xquard_r, LANGS
from transformers import AutoModel, AutoTokenizer

pd.set_option("display.max_colwidth", None)


def encode_sentences(sentences, model, tokenizer):
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        return (
            model(**encoded_input, return_dict=True)
            .pooler_output.cpu()
            .numpy()
            .reshape(-1)
        )


def get_relevant_docs(question, encoder_model, encoder_tokenizer, paragraphs_matrix):
    question_encoding = encode_sentences([question], encoder_model, encoder_tokenizer)
    question_encoding = torch.Tensor(question_encoding).cuda()
    scores = torch.nn.functional.cosine_similarity(paragraphs_matrix, question_encoding)
    return scores.argsort(descending=True)[:10].cpu().numpy()


def run_baseline():
    # Load the model
    encoder_modelname = "FacebookAI/xlm-roberta-base"
    encoder_model = AutoModel.from_pretrained(encoder_modelname).cuda()
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_modelname)

    # Load the dataset
    DATASETS = {lang: parse_xquard_r(lang) for lang in LANGS}

    concat_df = pd.concat(DATASETS.values(), axis=1)
    paragraphs = sum(
        [
            [row[f"paragraph_{lang}"] for i, row in concat_df.iterrows()]
            for lang in LANGS
        ],
        [],
    )
    unique_paragraphs = list(set(paragraphs))

    id_to_paragraph_dict = {
        i: paragraph for i, paragraph in enumerate(unique_paragraphs)
    }
    paragraph_to_id_dict = {
        paragraph: i for i, paragraph in enumerate(unique_paragraphs)
    }

    # Form the triplets
    triplets = [
        {
            "question": row[f"question_ar"],
            "positive_docs": [
                paragraph_to_id_dict[row[f"paragraph_{lang}"]] for lang in LANGS
            ],
        }
        for i, row in concat_df.iterrows()
    ]

    paragraphs_matrix = torch.Tensor(
        np.array(
            [
                encode_sentences([paragraph], encoder_model, encoder_tokenizer)
                for paragraph in tqdm(unique_paragraphs)
            ]
        )
    ).cuda()

    # Generate the predictions
    for triplet in tqdm(triplets):
        relevant_docs = get_relevant_docs(
            triplet["question"], encoder_model, encoder_tokenizer, paragraphs_matrix
        ).tolist()
        triplet["predicted_docs"] = relevant_docs
        triplet["is_relevant"] = [
            index in triplet["positive_docs"] for index in relevant_docs
        ]
        triplet["predicted_paragraphs"] = [
            id_to_paragraph_dict[i] for i in relevant_docs
        ]

    # Evaluate the model
    total_relevant_retrieved_docs = sum(
        [sum(triplet["is_relevant"]) > 0 for triplet in triplets]
    )
    total_queries_with_relevant_retrieved_docs = sum(
        [any(triplet["is_relevant"]) for triplet in triplets]
    )
    total_relevant_docs = sum([len(triplet["positive_docs"]) for triplet in triplets])
    print(
        f"% Relevant Paragraphs retrieved: {(total_relevant_retrieved_docs/total_relevant_docs)*100:.2f}%,"
        f" {total_relevant_retrieved_docs}/{total_relevant_docs}"
    )
    print(
        f"% Queries with Relevant Paragraphs retrieved: {(total_queries_with_relevant_retrieved_docs/len(triplets))*100:.2f}%,"
        f" {total_queries_with_relevant_retrieved_docs}/{len(triplets)}"
    )


if __name__ == "__main__":
    run_baseline()
