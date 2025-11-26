import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import os
from tqdm import tqdm

from load_data import ProteinDataset, analyze_dataset, collate_fn
from custom_transformer import SmallTransformer
import classifier

from argparse import Namespace

# -----------------------------
# CONFIG
# -----------------------------
MAX_LEN = 1024
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "./checkpoint.pt"
BEST_MODEL_PATH = "./best_model.pt"
PATIENCE = 5  # early stopping patience

# -----------------------------
# Model
# -----------------------------
class custom_classifier(nn.Module):
    def __init__(self,  args):
        super().__init__()
        self.args = args
        self.transformer = SmallTransformer(vocab_size = args.vocab_size,
                                                               attention_fn = args.attention_fn,
                                                                embed_dim = args.embed_dim, 
                                                                num_heads = args.num_heads,
                                                                depth = args.num_layers, 
                                                                max_len = args.max_len)
        self.classifier = args.Classifier(args)
        
    def forward(self, tokens, mask):
        x = self.transformer(tokens, mask)
        if self.args.classifier_reduction == "mean":
            clf_x = torch.mean(x, dim = 1) 
        else:
            clf_x = x[:,0,:]

        logits = self.classifier(clf_x)
        return logits

# -----------------------------
# Training loop with early stopping
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                start_epoch=0, epochs=50, patience=5, label_cols=None):

    model.to(DEVICE)
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch in train_loop:
            sequences, labels, attention_mask = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            logits = model(sequences, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            # Update tqdm postfix with current loss
            train_loop.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for batch in val_loop:
                sequences, labels, attention_mask = [b.to(DEVICE) for b in batch]
                logits = model(sequences, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())

                # Optional: update tqdm with batch loss
                val_loop.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss /= len(val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        preds_bin = (all_preds >= 0.5).float()
        macro_f1 = f1_score(all_labels, preds_bin, average="macro", zero_division=0)

        # Per-class metrics
        per_class_f1 = f1_score(all_labels, preds_bin, average=None, zero_division=0)
        per_class_auc = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except ValueError:
                auc = float('nan')
            per_class_auc.append(auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Macro F1: {macro_f1:.4f}")
        for i, label in enumerate(label_cols):
            print(f"  {label:20s} - F1: {per_class_f1[i]:.4f} - ROC-AUC: {per_class_auc[i]:.4f}")

        # Scheduler step
        if scheduler:
            scheduler.step(macro_f1)
            print("Current LR:", scheduler.get_last_lr())

        # Save checkpoint every epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_macro_f1": best_f1
        }
        torch.save(checkpoint, MODEL_CHECKPOINT)

        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model with Macro F1: {best_f1:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement in {patience} epochs, stopping early.")
                break

# -----------------------------
# Resume from checkpoint
# -----------------------------
def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=MODEL_CHECKPOINT):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_macro_f1 = checkpoint.get("best_macro_f1", 0.0)
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return model, optimizer, scheduler, start_epoch, best_macro_f1


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv("data/Swissprot_Train_Validation_dataset.csv")

    # The one-hot columns (based on your table)
    label_cols = [
        "Membrane", "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
        "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole",
        "Golgi apparatus", "Peroxisome"
    ]

    analyze_dataset(df, label_cols=label_cols)

    # First, split off test set (~10-20%)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    # Then split train/validation (~80/20 of remaining)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, shuffle=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = ProteinDataset(train_df, label_cols)
    val_dataset = ProteinDataset(val_df, label_cols)
    test_dataset = ProteinDataset(test_df, label_cols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    args = {
        "max_len" : MAX_LEN,
        "vocab_size": 23,
        "num_classes" : len(label_cols),
        "num_heads" : 2,
        "num_layers" : 2,
        "embed_dim" : 8,
        "attention_fn" : None,
        "Classifier" : classifier.LinearClassifier,
        "classifier_reduction" : None
    }

    args = Namespace(**args)

    # -----------------------------
    # Create model
    # -----------------------------
    model = custom_classifier(args)
    print(model)
    
    # -----------------------------
    # Loss and optimizer
    # -----------------------------
    n_train = len(train_df)
    pos = train_df[label_cols].sum().values
    neg = n_train - pos
    pos_weight = torch.tensor((neg / (pos + 1e-12)).astype(np.float32)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Scheduler: reduce LR if macro F1 stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    if False:
        model, optimizer, scheduler, start_epoch, best_f1 = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path=MODEL_CHECKPOINT
    )

    print(f"Found {DEVICE} for training!")

    train_model(model, test_loader, val_loader, criterion, optimizer, scheduler=scheduler,
            epochs=50, patience=5, label_cols=label_cols)

if __name__ == "__main__":
    main()