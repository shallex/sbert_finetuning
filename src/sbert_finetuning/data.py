import pandas as pd 
import random
import torch


class Dataset:
    def __init__(self, data_path, tokenizer):
        data = pd.read_csv(data_path)
        self.unique_labels = set(data["label"].unique())
        
        all_texts = data["text"].values.tolist()
        self.encodings = tokenizer(all_texts, truncation=True, padding=True, return_tensors='pt')
        self.labels = data["label"].values.tolist()

        self.lbl2idxs = {}
        for label in self.unique_labels:
            self.lbl2idxs[label] = set(data[data["label"] == label].index.values)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = self.labels[idx]
        anchor['labels'] = torch.tensor(label)

        pos_idxs = self.lbl2idxs[label] - {idx}
        pos_idx = random.sample(pos_idxs, k=1)[0]
        pos = {key: torch.tensor(val[pos_idx]) for key, val in self.encodings.items()}
        pos["labels"] = label

        neg_labels = self.unique_labels - {label}
        neg_label = random.sample(neg_labels, k=1)[0]
        neg_idxs = random.sample(self.lbl2idxs[neg_label], k=1)[0]
        neg_idx = random.sample(neg_idxs, k=1)[0]
        neg = {key: torch.tensor(val[neg_idx]) for key, val in self.encodings.items()}
        neg["labels"] = self.labels[neg_idx]

        return anchor, pos, neg
