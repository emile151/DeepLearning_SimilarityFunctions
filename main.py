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
        "vocab_size": 25,
        "num_classes" : 6,
        "attention_fn" : 'rbf',
        "num_heads" : 2,
        "num_layers" : 2,
        "embed_dim" : 4,
        "dropout" : 0.3,
        "Classifier" : classifier.LinearClassifier,
        "loss" : F.cross_entropy,
        "optimizer" : torch.optim.Adam,
        "num_epochs" : 10,
        "device" : device,
        "output_dir" : "/home/emile/PythonProjects/DeepLearning_SimilarityFunctions/results/",
        "experiment_title" : "small_embed_dim"
    }

    args = Namespace(**args)

    path_to_data = "data/complete_set_unpartitioned.fasta"
    batch_size = 32

    train_dataloader, dev_dataloader, test_dataloader = dataset.get_dataloaders(path_to_data, batch_size, args.max_len)
    model, evals = train.train_model(train_dataloader, dev_dataloader, args)
    preds, targets = train.test_model(model, test_dataloader, args)
    print(preds[0])
    print(targets[0])


if __name__ == "__main__":
    main()
