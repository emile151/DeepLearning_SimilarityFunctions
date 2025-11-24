import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import get_dataloaders
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

import custom_transformer




class SignalP(nn.Module):
    def __init__(self,  args):
        super().__init__()
        self.args = args
        self.transformer = custom_transformer.SmallTransformer(vocab_size = args.vocab_size,
                                                               attention_fn = args.attention_fn,
                                                                embed_dim = args.embed_dim, 
                                                                num_heads = args.num_heads,
                                                                depth = args.num_layers, 
                                                                max_len = args.max_len)
        self.classifier = args.Classifier(args)
        
    def forward(self, tokens):
        x = self.transformer(tokens)
        clf_x = x[:,0,:]
        logits = self.classifier(clf_x)
        return logits
    def forward_encode(self, tokens):
        x = self.transformer(tokens)
        return x

def run_epoch(model, data_loader, mode, args):
    loss_fn = args.loss
    optimizer = args.optimizer(model.parameters(), lr=1e-3)
    tqdm_bar = tqdm(data_loader, total=len(data_loader))

    if mode == 'Train':
        model.train()
    else:
        model.eval()
    batch_loss = 0
    i = 0
    preds = []
    targets = []
    losses = []
    for batch, (inputs, batch_targets) in enumerate(data_loader):
        inputs, batch_targets = inputs.to(args.device), batch_targets.to(args.device)
        batch_preds = model(inputs)
        loss = loss_fn(batch_preds, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        losses.append(loss.item())
        preds.append(batch_preds.detach().cpu())
        targets.append(batch_targets.detach().cpu())
        tqdm_bar.update()

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), np.mean(losses)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_auroc(pred_logits, targets):
    """
    pred_logits: (B, C) raw logits
    targets: (B,) integer class labels
    """

    pred = to_numpy(pred_logits)
    y = to_numpy(targets)

    # Multiclass: pass multi_class="ovr"
    if pred.shape[1] > 2:
        return roc_auc_score(y, pred, multi_class="ovr")
    else:
        # Binary: take probability of class 1
        probs = pred[:, 1]
        return roc_auc_score(y, probs)

def compute_auprc(pred_logits, targets):
    pred = to_numpy(pred_logits)
    y = to_numpy(targets)

    C = pred.shape[1]

    # Multiclass → compute AUPRC for each class (one-vs-rest)
    if C > 2:
        scores = []
        for c in range(C):
            y_bin = (y == c).astype(int)
            scores.append(average_precision_score(y_bin, pred[:, c]))
        return np.mean(scores)

    # Binary
    probs = pred[:, 1]
    return average_precision_score(y, probs)

def compute_mcc(pred_logits, targets):
    pred = to_numpy(pred_logits)
    y = to_numpy(targets)
    pred_labels = np.argmax(pred, axis=1)

    return matthews_corrcoef(y, pred_labels)

def eval(preds, targets):
    preds = F.softmax(preds, dim=1)
    auroc = compute_auroc(preds, targets)
    auprc = compute_auprc(preds, targets)
    mcc = compute_mcc(preds, targets)

    metrics = {
        "auroc" : auroc,
        "auprc" : auprc,
        "mcc"   : mcc
    }
    return metrics

def train_model(train_dataloader, dev_dataloader, args):
    model = SignalP(args).to(args.device)
    path_to_model = args.output_dir + args.experiment_title + ".pth"
    print("Model will be saved at: ", path_to_model)
    evals = {
        "Train" : [],
        "Dev" : [],
        "loss" : []
    }

    for epoch in range(args.num_epochs):
        print("Epoch = ", str(epoch + 1))
        for mode, data_loader in [('Train', train_dataloader),('Dev', dev_dataloader)]:
            print(mode, " for epoch ", str(epoch + 1))
            preds, targets, loss = run_epoch(model, data_loader, mode, args)
            epoch_eval = eval(preds, targets )
            evals[mode].append(epoch_eval)
            if mode == 'Train': 
                evals["loss"].append(loss.item())
            print("AUROC at epoch ", str(epoch + 1), " = ", epoch_eval["auroc"])
            print("AUPRC at epoch ", str(epoch + 1), " = ", epoch_eval["auprc"])
            print("MCC at epoch ", str(epoch + 1), " = ", epoch_eval["mcc"])
            print("Loss at epoch ", str(epoch + 1), " = ", loss)
            print("--------------------------------------------------------------------------")
        torch.save(model, path_to_model)
    return model, evals

def test_model(model, test_dataloader, args):
    preds, targets, loss = run_epoch(model, test_dataloader, 'Test', args)
    test_eval = eval(preds, targets)
    print("Test AUROC at epoch " , test_eval["auroc"])
    print("AUPRC at epoch ", test_eval["auprc"])
    print("MCC at epoch ", test_eval["mcc"])
    print("Loss at epoch ", loss)
    print("--------------------------------------------------------------------------")
    return preds, targets
