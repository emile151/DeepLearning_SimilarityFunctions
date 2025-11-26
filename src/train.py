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
        if self.args.classifier_reduction == "mean":
            clf_x = torch.mean(x, dim = 1) 
        else:
            clf_x = x[:,0,:]

        logits = self.classifier(clf_x)
        return logits
    def forward_encode(self, tokens):
        x = self.transformer(tokens)
        return x
    
    def forward_idx(self, tokens, idx):
        x = self.transformer.forward_block(tokens, idx)
        return x

def run_epoch(model, data_loader, mode, args):
    loss_fn = args.loss
    optimizer = args.optimizer(model.parameters(), lr=args.lr)
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

def run_epoch_embed(model, data_loader, mode, args):
    tqdm_bar = tqdm(data_loader, total=len(data_loader))
    model.eval()
    batch_loss = 0
    i = 0
    preds = []
    targets = []
    losses = []
    for batch, (inputs, batch_targets) in enumerate(data_loader):
        inputs, batch_targets = inputs.to(args.device), batch_targets.to(args.device)
        batch_preds = model.forward_encode(inputs)
        preds.append(batch_preds.detach().cpu())
        targets.append(batch_targets.detach().cpu())
        tqdm_bar.update()

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)

def run_epoch_idx(model, data_loader, args, idx):
    tqdm_bar = tqdm(data_loader, total=len(data_loader))
    model.eval()
    preds = []
    targets = []
    for batch, (inputs, batch_targets) in enumerate(data_loader):
        inputs, batch_targets = inputs.to(args.device), batch_targets.to(args.device)
        batch_preds = model.forward_idx(inputs, idx)
        preds.append(batch_preds.detach().cpu())
        targets.append(batch_targets.detach().cpu())
        tqdm_bar.update()

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)
def compute_auroc(pred_logits, targets, multilabel=False):
    y = to_numpy(targets)
    
    if multilabel:
        # Multilabel: sigmoid + compute per class
        pred_probs = to_numpy(torch.sigmoid(pred_logits))
        scores = []
        for c in range(y.shape[1]):
            scores.append(roc_auc_score(y[:, c], pred_probs[:, c]))
        return np.mean(scores)
    
    else:
        # Binary / multiclass: softmax
        pred_probs = to_numpy(F.softmax(pred_logits, dim=1))
        C = pred_probs.shape[1]
        if C > 2:
            return roc_auc_score(y, pred_probs, multi_class="ovr")
        else:
            # Binary
            return roc_auc_score(y, pred_probs[:, 1])

def compute_auprc(pred_logits, targets, multilabel=False):
    y = to_numpy(targets)
    
    if multilabel:
        pred_probs = to_numpy(torch.sigmoid(pred_logits))
        scores = []
        for c in range(y.shape[1]):
            scores.append(average_precision_score(y[:, c], pred_probs[:, c]))
        return np.mean(scores)
    
    else:
        pred_probs = to_numpy(F.softmax(pred_logits, dim=1))
        C = pred_probs.shape[1]
        if C > 2:
            scores = []
            for c in range(C):
                y_bin = (y == c).astype(int)
                scores.append(average_precision_score(y_bin, pred_probs[:, c]))
            return np.mean(scores)
        else:
            return average_precision_score(y, pred_probs[:, 1])

def compute_mcc(pred_logits, targets, multilabel=False):
    y = to_numpy(targets)
    
    if multilabel:
        pred_probs = to_numpy(torch.sigmoid(pred_logits))
        scores = []
        for c in range(y.shape[1]):
            pred_labels = (pred_probs[:, c] > 0.5).astype(int)
            scores.append(matthews_corrcoef(y[:, c], pred_labels))
        return np.mean(scores)
    
    else:
        pred_probs = to_numpy(F.softmax(pred_logits, dim=1))
        pred_labels = np.argmax(pred_probs, axis=1)
        return matthews_corrcoef(y, pred_labels)

def eval(pred_logits, targets, multilabel=False):
    metrics = {
        "auroc": compute_auroc(pred_logits, targets, multilabel),
        "auprc": compute_auprc(pred_logits, targets, multilabel),
        "mcc": compute_mcc(pred_logits, targets, multilabel)
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
        for mode, data_loader in [('Train', train_dataloader)]:#,('Dev', dev_dataloader)]:
            print(mode, " for epoch ", str(epoch + 1))
            preds, targets, loss = run_epoch(model, data_loader, mode, args)
            epoch_eval = eval(preds, targets, args.is_multilabel)
            evals[mode].append(epoch_eval)
            if mode == 'Train': 
                evals["loss"].append(loss.item())
            print(mode + "_AUROC at epoch ", str(epoch + 1), " = ", epoch_eval["auroc"])
            print(mode + "_AUPRC at epoch ", str(epoch + 1), " = ", epoch_eval["auprc"])
            print(mode + "_MCC at epoch ", str(epoch + 1), " = ", epoch_eval["mcc"])
            print(mode + "_Loss at epoch ", str(epoch + 1), " = ", loss)
            if mode == "Train" and loss < 0.01:
                break;
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

def test_model_embed(model, test_dataloader, args):
    preds, targets = run_epoch_embed(model, test_dataloader, 'Test', args)
    #print("Test AUROC at epoch " , test_eval["auroc"])
    #print("AUPRC at epoch ", test_eval["auprc"])
    #print("MCC at epoch ", test_eval["mcc"])
    #print("Loss at epoch ", loss)
    print("--------------------------------------------------------------------------")
    return preds, targets

def test_model_idx(model, test_dataloader, args, idx):
    preds, targets = run_epoch_idx(model, test_dataloader, args, idx)

    print("--------------------------------------------------------------------------")
    return preds, targets