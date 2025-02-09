## Zero-shot retrieval of multilingual passages using XQUAD-r
1. Downloading the datasets: `python download_datasets.py`
2. Running the zero-shot xlm-roberta baseline: `python run_baseline.py`

#### Experiment
- Setup: questions in Arabic, retrieve paragraphs in any of the languages.
- Finding: `language bias` indeed exists, where the returned paragraph are in Arabic, and non-Arabic paragraph are rarely retrieved.