import torch
import torch.nn.functional as F
import sys
import math

from argparse import Namespace


sys.path.append("src") 

import classifier
import dataset
import train

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = {
        "max_len" : 72,
        "vocab_size": 26,
        "num_classes" : 3,
        "num_heads" : 2,
        "num_layers" : 2,
        "embed_dim" : 8,
        "attention_fn" : None,
        "dropout" : 0.2,
        "Classifier" : classifier.LinearClassifier,
        "loss" : F.cross_entropy,
        "optimizer" : torch.optim.Adam,
        "lr" : 1e-4,
        "num_epochs" : 30,
        "device" : device,
        "output_dir" : "/home/emile/PythonProjects/DeepLearning_SimilarityFunctions/results/",
        "experiment_title" : "test_average_linear",
        "classifier_reduction" : "mean",
        "exclude_class" : ["TA", "PILI", "TATLIP"]
    }

    args = Namespace(**args)

    path_to_data = "data/dataset.csv"
    batch_size = 32

    print(f"Train on device: {device}")
    train_dataloader, dev_dataloader, test_dataloader = dataset.get_dataloaders(path_to_data, batch_size, args.max_len, args.exclude_class, use_sample_weights = False)
    model, evals = train.train_model(train_dataloader, dev_dataloader, args)
    preds, targets = train.test_model(model, test_dataloader, args)



if __name__ == "__main__":
    main()
