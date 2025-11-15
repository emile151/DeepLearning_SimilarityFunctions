import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

import sys
sys.path.append("src")

import dataset
import custom_transformer
import classifier


path_to_data = "data/complete_set_unpartitioned.fasta"
max_len = 72
batch_size = 32
vocab_size = 25
num_classes = 6
attention_fn = custom_transformer.attention
num_heads = 4
num_layers = 2
embed_dim = 24
dropout = 0

train_dataloader, test_dataloader = dataset.get_dataloaders(path_to_data, batch_size, max_len)



class SignalP(nn.Module):
    def __init__(self,  max_len, vocab_size, num_classes, attention_fn, num_heads, num_layers, embed_dim, dropout):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.attention_fn = attention_fn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.transformer = custom_transformer.SmallTransformer(vocab_size = vocab_size,
                                                               attention_fn = attention_fn,
                                                                embed_dim = embed_dim, 
                                                                num_heads = num_heads,
                                                                depth = num_layers, 
                                                                max_len = max_len)
        self.classifier = classifier.LinearClassifier(embed_dim, num_classes)
        
    def forward(self, tokens):
        x = self.transformer(tokens)
        print("X: ", x.shape)

        clf_x = x[:,0,:]
        print("X: ", clf_x.shape)
        logits = self.classifier(clf_x)
        return logits






model = SignalP(max_len = max_len, vocab_size = vocab_size, num_classes = num_classes, attention_fn = attention_fn, num_heads = num_heads, num_layers = num_layers, embed_dim = embed_dim, dropout = dropout)
loss_fn = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()


running_loss = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch, (inputs, targets) in enumerate(train_dataloader):
    inputs, targets = inputs.to(device), targets.to(device)
    print("input shape", inputs.shape)
    print("input shape", inputs.shape)
    preds = model(inputs)
    print(preds.shape)
    loss = loss_fn(preds, targets)
    print("loss", loss)
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)