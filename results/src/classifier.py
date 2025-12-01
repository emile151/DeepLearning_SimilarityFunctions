import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, args):
        super(LinearClassifier, self).__init__()
        self.args = args
        self.classifier = nn.Linear(args.embed_dim, args.num_classes)

    def forward(self, encoding):
        logits = self.classifier(encoding)
        return logits