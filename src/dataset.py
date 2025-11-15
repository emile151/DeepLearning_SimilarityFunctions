import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class SignalPeptides(Dataset):
    def __init__(self, path_to_dataset):
        self.data = create_pandas_df_from_path(path_to_dataset)
        self.path_to_dataset = path_to_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx,4],self.data.iloc[idx,3]


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
            sequence.append(line)
        else:
            cell_location.append(line)
        c += 1
    
    unique_classes = sorted(set(class_nam))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    int_labels = [class_to_idx[nam] for nam in class_nam]

    data = pd.DataFrame({
        "id" : ids,
        "domain" : domain,
        "class" : class_nam,
        "class_ints" : int_labels,
        "sequence" : sequence,
        "cell_location" : cell_location
    })
    return data

def get_dataloaders(path_to_data, batch_size = 32):
    data = create_pandas_df_from_path(path_to_data)
    train_df, test_df = train_test_split(
        data, 
        test_size=0.2, 
        shuffle=True, 
        random_state=42
    )
    train_dataloader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_df, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader