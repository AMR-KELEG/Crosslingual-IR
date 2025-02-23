# Setup?
# Question in Arabic -> retrieve documents in Arabic and English!

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from run_baseline import encode_sentences, get_relevant_docs

tqdm.pandas()
pd.set_option("display.max_colwidth", None)


def load_file_from_zip(filename):
    with ZipFile(filename, "r") as zip_file:
        with zip_file.open("chunk_0_0.jsonl", "r") as f:
            lines = [json.loads(l)["text"] for l in f]
    return lines


def load_data():
    data_dir = Path("data/fineweb_edu")

    # Load the data
    ar_data = load_file_from_zip(data_dir / "chunk_0_0_ar.zip")
    en_data = load_file_from_zip(data_dir / "chunk_0_0_en.zip")

    # Create a DataFrame
    df = pd.DataFrame({"ar": ar_data, "en": en_data})
    for lang in ["ar", "en"]:
        df[f"n_tokens{lang}"] = df[lang].progress_apply(lambda x: len(x.split()))
    return df


def main():
    # Load the data and align the paragraphs
    df = load_data()
    paragraphs = [[row["ar"], row["en"]] for i, row in df.iterrows()]
    paragraphs = [p for l in paragraphs for p in l]

    # Load the model
    encoder_modelname = "FacebookAI/xlm-roberta-base"
    encoder_model = AutoModel.from_pretrained(encoder_modelname).cuda()
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_modelname)

    # Note: batching the encoding process did not significantly speed up the process
    # TODO: Check if FastTokenizer can be used to speed up the process!
    BATCH_SIZE = 32

    encodings = [
        np.array(
            encode_sentences(
                paragraphs[i : i + BATCH_SIZE], encoder_model, encoder_tokenizer
            )
        )
        for i in trange(0, len(paragraphs[:10000]), BATCH_SIZE, desc="Encoding")
    ]

    encodings_matrix = torch.Tensor(np.concatenate(encodings, axis=0)).cuda()

    # Test the model on a random question
    question = "معلومات أكثر عن مرض السكري"
    question_encoding = torch.Tensor(
        encode_sentences([question], encoder_model, encoder_tokenizer)
    ).cuda()

    similarity_scores = torch.nn.functional.cosine_similarity(
        encodings_matrix, question_encoding
    )

    retrieved_df = pd.DataFrame(
        {
            "indecies": torch.topk(similarity_scores.flatten(), 3)
            .indices.cpu()
            .numpy(),
            "scores": torch.topk(similarity_scores.flatten(), 3).values.cpu().numpy(),
        }
    )
    retrieved_df["paragraph"] = retrieved_df["indecies"].apply(lambda x: paragraphs[x])

    # High similarity score even for the lowest ranked paragraph!
    print(similarity_scores.min())

    # Apply constrastive learning on the paragraphs
    BATCH_SIZE = 16
    dataset = paragraphs[:128]

    encoder_model.train()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)
    for epoch in trange(10, desc="Epochs"):
        indecies = [
            i
            for i in range(
                0,
                len(dataset) // 2,
            )
        ]
        np.random.shuffle(indecies)
        for i in trange(0, len(indecies), BATCH_SIZE, desc="Batches"):
            batch = [
                [dataset[2 * i], dataset[2 * i + 1]]
                for i in indecies[i : i + BATCH_SIZE]
            ]
            batch = [p for l in batch for p in l]
            batch_tokenized = encoder_tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")

            # 16 sentence pairs!
            batch_encodings = encoder_model(**batch_tokenized).pooler_output
            simialrities = batch_encodings @ batch_encodings.T

            # Compute the cross entropy loss
            batch_loss = loss(simialrities, torch.arange(simialrities.shape[0]).cuda())
            print(batch_loss)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(indecies)

if __name__ == "__main__":
    main()
