import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, hidden_dims, num_classes):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dims, num_classes)
        

    def forward(self, embedding):
        logits = self.classifier(embedding)
        return logits