import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append("src")

import dataset
import custom_transformer
import classifier
import train

from argparse import Namespace

path_to_data = "/home/emile/PythonProjects/DeepLearning_SimilarityFunctions/data/Swissprot_Train_Validation_dataset_with_onehot.csv"
max_len = 72
batch_size = 32
vocab_size = 25
num_classes = 6
attention_fn = None #custom_transformer.attention
num_heads = 4
num_layers = 2
embed_dim = 24
dropout = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

args = {
            "max_len" : 300,
            "vocab_size": 26,
            "num_classes" : 10,
            "num_heads" : 8,
            "num_layers" : 4,
            "embed_dim" : 128,
            "attention_fn" : "rbf",
            "dropout" : 0.2,
            "Classifier" : classifier.LinearClassifier,
            "loss" : F.cross_entropy,
            "optimizer" : torch.optim.Adam,
            "lr" : 1e-4,
            "num_epochs" : 1000,
            "device" : device,
            "output_dir" : "/zhome/01/3/213912/DeepLearning_SimilarityFunctions/results",
            "experiment_title" : "prtstruct_linear_attn",
            "classifier_reduction" : "mean",
            "exclude_class" : [],
            "is_multilabel": False
    }


    

args = Namespace(**args)

path_to_data = "/home/emile/PythonProjects/DeepLearning_SimilarityFunctions/data/protein_struct_classification.csv"
batch_size = 32

model = torch.load('/home/emile/PythonProjects/DeepLearning_SimilarityFunctions/results/prtstruct_rbf_attn.pth', weights_only = False, map_location=torch.device('cpu'))
train_dataloader, dev_dataloader, test_dataloader = dataset.get_dataloaders(path_to_data, batch_size, args.max_len, args.is_multilabel, args.exclude_class, use_sample_weights = False)

del train_dataloader
del dev_dataloader
print(model)

embeds, targets = train.test_model_embed(model, test_dataloader, args)
embeds = torch.mean(embeds, dim = 1)


#col = torch.argmax(embeds, dim=1)
print(embeds.shape)
pca = PCA()
embeds_pca = pca.fit_transform(embeds)
print("Variance explained: ",pca.fit(embeds).explained_variance_ratio_)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embeds_pca[:,0], embeds_pca[:,1], embeds_pca[:,2], c = targets, cmap = "tab10" )
plt.legend()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embeds_pca[:,2], embeds_pca[:,0], embeds_pca[:,1], c = targets, cmap = "tab10" )
plt.legend()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embeds_pca[:,1], embeds_pca[:,2], embeds_pca[:,0], c = targets, cmap = "tab10" )
plt.legend()
plt.show()