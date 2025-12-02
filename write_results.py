import torch
import torch.nn.functional as F
import sys
import math
import os

from argparse import Namespace

ROOT = os.path.dirname(os.path.abspath(__file__))   # directory of main.py
SRC = os.path.join(ROOT, "src")

sys.path.append(SRC)

import classifier
import dataset
import train

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For multilabel use F.binary_cross_entropy_with_logits
    # num_classes = 11
    # is_multilabel = True
    args = {
            "max_len" : 72,
            "vocab_size": 26,
            "num_classes" : 6,
            "num_heads" : 8,
            "num_layers" : 2,
            "embed_dim" : 128,
            "attention_fn" : None,
            "dropout" : 0.2,
            "Classifier" : classifier.LinearClassifier,
            "loss" : F.cross_entropy,
            "optimizer" : torch.optim.Adam,
            "lr" : 1e-4,
            "num_epochs" : 1000,
            "device" : device,
            "output_dir" : "/zhome/01/3/213912/DeepLearning_SimilarityFunctions/results/",
            "experiment_title" : "signalp_linear_attn",
            "classifier_reduction" : "mean",
            "exclude_class" : [],
            "is_multilabel": False
    }

    args = Namespace(**args)

    path_to_data = "/zhome/01/3/213912/DeepLearning_SimilarityFunctions/data/dataset.csv"
    batch_size = 32
    model = torch.load('/zhome/01/3/213912/DeepLearning_SimilarityFunctions/results/signalp_linear_attn.pth', weights_only = False, map_location=torch.device(args.device))
    print(f"Test on device: {device}")
    train_dataloader, dev_dataloader, test_dataloader = dataset.get_dataloaders(path_to_data, batch_size, args.max_len, args.is_multilabel, args.exclude_class, use_sample_weights = False)
    
    del train_dataloader
    del dev_dataloader

    model.eval()
    batch_loss = 0
    i = 0
    preds = []
    targets = []
    losses = []
    embeds = []
    for batch, (inputs, batch_targets) in enumerate(test_dataloader):
        inputs, batch_targets = inputs.to(args.device), batch_targets.to(args.device)
        batch_preds, emb = model.forward_encode(inputs)
        preds.append(batch_preds.detach().cpu())
        embeds.append(emb)
        targets.append(batch_targets.detach().cpu())

    logits = torch.cat(preds, dim=0)
    embeds = torch.cat(embeds, dim=0)
    targets = torch.cat(targets, dim=0)
    print("Targets: ", targets.shape)
    print("Targets: ", targets)
    targets = targets.unsqueeze(1)
    concat_tensor = torch.cat([logits, embeds, targets], dim=1)
    torch.save({
    "all_labels": targets,
    "predictions": logits,
    "embeddings": embeds}, "/zhome/01/3/213912/DeepLearning_SimilarityFunctions/results/signalp_linear_results.pt")
    
if __name__ == "__main__":
    main()
