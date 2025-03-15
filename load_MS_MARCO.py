import datasets
from tqdm import tqdm
from torch.utils.data import Dataset


class MS_MARCO(Dataset):
    def __init__(self, split, answers_only, version="v1.1"):
        assert split in ["train", "validation", "test"]
        dataset_name = "microsoft/ms_marco"
        self.split = split

        dataset = datasets.load_dataset(dataset_name, version, split=split)
        df = dataset.to_pandas()

        # Extract the queries and relevant documents
        self.queries = df["query"]
        if answers_only:
            self.docs = df["answers"].apply(lambda l: l.tolist())
        else:
            self.docs = df["passages"].apply(lambda d: d["passage_text"].tolist())

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.docs[idx]


if __name__ == "__main__":
    dataset = MS_MARCO("validation", answers_only=True)
    print(dataset.__getitem__(0))
