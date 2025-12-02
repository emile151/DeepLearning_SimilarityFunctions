import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

class SignalPeptides(Dataset):
    def __init__(self, data, max_len, multilabel = False):
        self.data = data.reset_index(drop=True)
        self.seqs = self.data['sequence']
        self.labels = self.data['class_ints']
        self.max_len = max_len
        self.multilabel = multilabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.seqs.iloc[idx]
        label = self.labels.iloc[idx]
        aas = ["[CLS]","[PAD]","A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V","B","Z","X","U"]
        vocab = {aa: idx for idx, aa in enumerate(aas)}
        # tokenize sequence
        token_seq = [vocab["[PAD]"]] * self.max_len
        token_seq[0] = vocab["[CLS]"]

        for pos, aa in enumerate(seq[:self.max_len-1], start=1):
            token_seq[pos] = vocab.get(aa, vocab["X"])

        # convert to tensor
        token_seq = torch.tensor(token_seq, dtype=torch.long)
        if self.multilabel:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)
        return token_seq, label

def tokenize(seqs, max_len):
    aas = ["[CLS]","[PAD]","A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V","B","Z","X","U"]
    vocab = {aa: idx for idx, aa in enumerate(aas)}
    tokenized_seqs = []
    for seq in seqs:
        token_seq = [vocab['[PAD]']] * max_len
        token_seq[0] = vocab['[CLS]']
        pos = 1
        for aa in seq:
            token_seq[pos] = vocab[aa]
            pos += 1
        tokenized_seqs.append(torch.tensor(token_seq))
    return tokenized_seqs

def create_pandas_df_from_path(path_to_dataset):
    data_lines = open(path_to_dataset, "r")
    ids = []
    domain = []
    class_nam = []
    sequence = []
    cell_location = []
    c = 0
    for line in data_lines:
        if c % 3 == 0:
            split = line.split("|")
            ids.append(split[0][1:])
            domain.append(split[1])
            class_nam.append(split[2][:-2])
        elif c % 3 == 1:
            sequence.append(line[:-2])
        else:
            cell_location.append(line[:-2])
        c += 1
    
    unique_classes = sorted(set(class_nam))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    print("class_to_idx", class_to_idx)
    int_labels = [torch.tensor(class_to_idx[nam]) for nam in class_nam]
    #tokenized_seqs = tokenize(sequence, max_len)
    data = pd.DataFrame({
        "id" : ids,
        "domain" : domain,
        "class" : class_nam,
        "class_ints" : int_labels,
        "sequence" : sequence,
        #"tokenized_seqs": tokenized_seqs,
        "cell_location" : cell_location
    })
    return data

def get_dataloaders(path_to_data, batch_size = 32, max_len = 72, is_multilabel = False, exclude_class = [], use_sample_weights = True):
    if is_multilabel:
        df = pd.read_csv(path_to_data)
        df = df[df['sequence'].str.len() <= 100]
        df["class_ints"] = list(df.iloc[:, 4:15].values)
        train_df = df[df["split_group"] == "train"].reset_index(drop=True)
        dev_df   = df[df["split_group"] == "dev"].reset_index(drop=True)
        test_df  = df[df["split_group"] == "test"].reset_index(drop=True)
    
    elif path_to_data.split(".")[-1] == "fasta":
        data = create_pandas_df_from_path(path_to_data)
        data = data[~data["class"].isin(exclude_class)]
        train_df, test_df = train_test_split(
            data, 
            test_size=0.15, 
            shuffle=True, 
            random_state=42
        )
        dev_rel = 0.15 / 0.85
        train_df, dev_df = train_test_split(
            train_df, 
            test_size=dev_rel, 
            shuffle=True, 
            random_state=42
        )
        train_df["split_group"] = "train"
        dev_df["split_group"] = "dev"
        test_df["split_group"] = "test"

        combined = pd.concat([train_df, dev_df, test_df], ignore_index=True)
        combined["class_ints"] = combined["class_ints"].astype(int)
        path = "/".join(path_to_data.split("/")[0:-1]) + "/dataset.csv"
        combined.to_csv(path, index=False)
        print("CSV dataset written to: ", path)
    elif path_to_data.split(".")[-1] == "csv":
        df = pd.read_csv(path_to_data)
        df = df[~df["class"].isin(exclude_class)]
        class_nam = df["class"]
        unique_classes = sorted(set(class_nam))
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        print("class_to_idx", class_to_idx)
        int_labels = [torch.tensor(class_to_idx[nam]) for nam in class_nam]
        df["class_ints"] = int_labels
        train_df = df[df["split_group"] == "train"].reset_index(drop=True)
        dev_df   = df[df["split_group"] == "dev"].reset_index(drop=True)
        test_df  = df[df["split_group"] == "test"].reset_index(drop=True)

    train_set = SignalPeptides(train_df, max_len, multilabel=is_multilabel)
    dev_set = SignalPeptides(dev_df, max_len, multilabel=is_multilabel)
    test_set = SignalPeptides(test_df, max_len, multilabel=is_multilabel)

    if use_sample_weights:
        targets = np.array(train_set.labels)
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        
        print(class_sample_count)

        weight = 1. / class_sample_count
        print(weight)
        #weight = [1/6, 1/6, 1/2, 1/6, 1/2, 1/2]
        samples_weight = np.array([weight[int(t)] for t in targets])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement = True)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    dev_dataloader = DataLoader(dev_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)    
    return train_dataloader,dev_dataloader, test_dataloader
    
