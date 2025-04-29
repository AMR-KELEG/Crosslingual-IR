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
        df = df[df["query_type"] == "description"]

        # Filter out the queries with no answers
        df = df[df["answers"].apply(lambda l: len(l) > 0)]

        df.reset_index(drop=True, inplace=True)

        # Extract the queries and relevant documents
        self.queries = df["query"]
        if answers_only:
            self.relevant_docs = df["answers"].apply(lambda l: l.tolist())
        else:
            self.relevant_docs = df["passages"].apply(
                lambda d: d["passage_text"].tolist()
            )

        # Flatten the documents
        self.docs = sorted(
            set([doc for doc_list in self.relevant_docs for doc in doc_list])
        )
        self.doc_ids = {doc: i for i, doc in enumerate(self.docs)}
        self.relevant_ids = [
            [self.doc_ids[doc] for doc in doc_list] for doc_list in self.relevant_docs
        ]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.relevant_docs[idx]


if __name__ == "__main__":
    dataset = MS_MARCO("validation", answers_only=True)
    print(dataset.__getitem__(0))
