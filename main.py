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
            "num_heads" : 1,
            "num_layers" : 2,
            "embed_dim" : 4,
            "attention_fn" : None,
            "dropout" : 0.2,
            "Classifier" : classifier.LinearClassifier,
            "loss" : F.cross_entropy,
            "optimizer" : torch.optim.Adam,
            "lr" : 1e-4,
            "num_epochs" : 1,
            "device" : device,
            "output_dir" : "results/",
            "experiment_title" : "test_scripts",
            "classifier_reduction" : "mean",
            "exclude_class" : [],
            "is_multilabel": False
    }

    args = Namespace(**args)

    path_to_data = "/home/emile/PythonProjects/DeepLearning_SimilarityFunctions/data/dataset.csv"
    batch_size = 32

    print(f"Train on device: {device}")
    train_dataloader, dev_dataloader, test_dataloader = dataset.get_dataloaders(path_to_data, batch_size, args.max_len, args.is_multilabel, args.exclude_class, use_sample_weights = False)
    model, evals = train.train_model(train_dataloader, dev_dataloader, args)
    preds, targets, embeds = train.test_model(model, test_dataloader, args)



if __name__ == "__main__":
    main()
