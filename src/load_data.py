import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

def encode_and_fix(seq, max_len=1024):
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    # 0 = PAD, 1 = CLS, 2+ = amino acids, X = unknown
    aa_to_idx = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
    aa_to_idx["X"] = len(aa_to_idx) + 2

    CLS_TOKEN = 1

    # Encode sequence
    encoded = [aa_to_idx.get(aa, aa_to_idx["X"]) for aa in seq]

    # Prepend CLS token
    encoded = [CLS_TOKEN] + encoded

    # Truncate from the end → keep N-terminus
    if len(encoded) > max_len:
        encoded = encoded[:max_len]

    # Pad
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))

    return torch.tensor(encoded, dtype=torch.long)

class ProteinDataset(Dataset):
    def __init__(self, df, label_cols, seq_col="Sequence", max_len=1024):
        self.sequences = df[seq_col].tolist()
        self.labels = df[label_cols].values.astype("float32")
        self.length = max_len
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_encoded = encode_and_fix(self.sequences[idx], max_len=self.length)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq_encoded, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    # Pad sequences
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Stack labels
    labels = torch.stack(labels)
    
    # Attention mask: 1 for real tokens, 0 for PAD
    attention_mask = (padded_sequences != 0).long()
    
    return padded_sequences, labels, attention_mask

def analyze_dataset(df, sequence_col="Sequence", label_cols=None):
    print("===== DATASET SUMMARY =====")
    
    # --- Basic shape ---
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print()

    # --- Sequence lengths ---
    seq_lengths = df[sequence_col].apply(len)

    print("===== SEQUENCE LENGTH STATISTICS =====")
    print(f"Min length:   {seq_lengths.min()}")
    print(f"Max length:   {seq_lengths.max()}")
    print(f"Mean length:  {seq_lengths.mean():.2f}")
    print(f"Median length:{seq_lengths.median():.2f}")
    print()

    # --- Label distribution ---
    if label_cols is not None:
        print("===== LABEL DISTRIBUTION =====")
        label_sum = df[label_cols].sum().sort_values(ascending=False)
        for label, count in label_sum.items():
            print(f"{label:25s}: {int(count)}")
        print()

    print("===== DONE =====")

def main():
    df = pd.read_csv("data/Swissprot_Train_Validation_dataset.csv")

    # The one-hot columns (based on your table)
    label_cols = [
        "Membrane", "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
        "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole",
        "Golgi apparatus", "Peroxisome"
    ]

    df["Encoded"] = df["Sequence"].apply(encode_and_fix)

    analyze_dataset(df, label_cols=label_cols)

    # Split into train & test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = ProteinDataset(train_df, label_cols)
    test_dataset  = ProteinDataset(test_df, label_cols)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    for seqs, labels, attn_mask in train_loader:
        print(seqs.shape)        # (batch_size, max_len)
        print(seqs)
        print(labels.shape)      # (batch_size, num_labels)
        print(attn_mask.shape)   # (batch_size, max_len)
        print(attn_mask)
        break

if __name__ == "__main__":
    main()