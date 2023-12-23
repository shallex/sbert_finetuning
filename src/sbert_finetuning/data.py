import pandas as pd 
import random
from sentence_transformers import SentencesDataset
from sentence_transformers.readers import InputExample


class Dataset:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.inputs = []
        for _, row in self.data.iterrows():
            self.inputs.append(InputExample(texts=[row["text"]], label=row["label"]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]
