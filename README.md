## Answer/Passage retrieval using MS-MARCO
The MS-MARCO dataset has queries (from user bing queries), relevant passages, and the answers. The queries' answer types are taxonomized into: description, numeric, entity, location, person. I focus on the `description` queries since they are more relevant to an exploratory knowledge seeking, the application I can see as useful for CrossLingual IR (CLIR).

#### Experiment (1)
- Setup: retrieve the correct answer(s) for each query.

#### Notes
- The evaluation metric `MRR@10` follows the metric used in the official passage-retrieval leaderboard: https://github.com/microsoft/MSMARCO-Passage-Ranking/blob/master/ms_marco_eval.py#L107

## Zero-shot retrieval of multilingual passages using XQUAD-r
1. Downloading the datasets: `python download_datasets.py`
2. Running the zero-shot xlm-roberta baseline: `python run_baseline.py`

#### Experiment
- Setup: questions in Arabic, retrieve paragraphs in any of the languages.
- Finding: `language bias` indeed exists, where the returned paragraph are in Arabic, and non-Arabic paragraph are rarely retrieved.

## Cross-lingual retrieval
1. Make sure you have access to the following dataset on HF `kaust-generative-ai/fineweb-edu-ar`
2. Generate an access token through: https://huggingface.co/settings/tokens
3. Download the parallel Arabic/English arXiv paragraphs after setting the access token as an environment variable
```
export HF_TOKEN="REPLACE_THIS_WITH_YOUR_ACCESS_TOKEN!"

wget --header="Authorization: Bearer ${HF_TOKEN}" -c "https://huggingface.co/datasets/kaust-generative-ai/fineweb-edu-ar/resolve/main/ar/train/chunk_0_0.zip" -O "chunk_0_0_ar.zip"
wget --header="Authorization: Bearer ${HF_TOKEN}" -c "https://huggingface.co/datasets/kaust-generative-ai/fineweb-edu-ar/resolve/main/en/train/chunk_0_0.zip" -O "chunk_0_0_en.zip"

DATA_DIR="data/fineweb_edu"
mkdir -p ${DATA_DIR}
mv chunk_0_0_{ar,en}.zip ${DATA_DIR}
```
4. Run the zero-shot xlm-roberta baseline `python fineweb_edu_retrieval.py`

#### Experiment
- Setup: question in Arabic or English, retrieve `fineweb_edu` documents in any of the two languages.
- Findings:
    - The similarity scores are really high
        - This might be a reason of the long length of the paragraphs
        - TODO: Split paragraphs into smaller segments (initially in a random way?)
        - TODO: How to perform alignment if such segmentation is done!
    - Language bias still exists!
    - **Preliminary Result (Qualitiative analysis)** Naive alignment of the parallel documents' embeddings using contrastive learning on a small batch of data detiriorates the quality of the retrieved documents.
