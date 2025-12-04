from sklearn.metrics import roc_curve,auc
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.patches as mpatches


def evaluation_plots(data, kernel, label_cols):
    all_labels = data["all_labels"].squeeze(1)
    all_preds = data["predictions"]
    all_preds = torch.softmax(data["predictions"], dim=1).cpu()
    all_emb = data["embeddings"].cpu().detach().numpy()
    all_labels_int = data["all_labels"].squeeze(1).cpu() # shape (N,)

    all_labels_int = all_labels_int.numpy()
    all_labels = torch.nn.functional.one_hot(all_labels, num_classes=6)
    values, counts = torch.unique(all_labels, return_counts=True)
    print(counts)
    true_predicted = [0,0,0,0,0,0]
    false_predicted = [0,0,0,0,0,0]

    for pred, lab in zip(all_preds, all_labels):
        p = np.argmax(pred)
        l = np.argmax(lab)
        if p == l:
            true_predicted[p] += 1
        else:
            false_predicted[p] += 1
    class_accs = np.array(true_predicted) / (np.array(false_predicted) + np.array(true_predicted))
    print("CLass accuracy: ", class_accs)
    print("Mean accuracy: ", np.mean(class_accs))
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    plt.bar(x = label_cols, height = np.array(true_predicted) / (np.array(false_predicted) + np.array(true_predicted)), color = colors)
    plt.title(kernel + " Class accuracies")
    plt.show()

    num_classes = all_preds.shape[1]

    per_class_mcc = []

    for cls in range(num_classes):
        y_true_binary = []
        y_pred_binary = []

        for pred, lab in zip(all_preds, all_labels):
            p = np.argmax(pred)
            l = np.argmax(lab)

            y_pred_binary.append(1 if p == cls else 0)
            y_true_binary.append(1 if l == cls else 0)

        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        per_class_mcc.append(mcc)

    per_class_mcc = np.array(per_class_mcc)

    print("Per-class MCC:", per_class_mcc)
    print("Mean MCC:", np.mean(per_class_mcc))

    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
    plt.bar(x=label_cols, height=per_class_mcc, color=colors)
    plt.title("Per-class MCC")
    plt.ylim(0.0,1.0)
    plt.ylabel("MCC")
    plt.show()

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i,lab in enumerate(label_cols):
        plt.plot(
            fpr[i], tpr[i],
            label=f"Class {lab} ROC (AUC = {roc_auc[i]:.2f})",
            color = colors[i]
        )

    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})",
        linestyle="--",
        color ="pink"
    )

    plt.plot([0, 1], [0, 1], "k--")

    plt.title(kernel + " Multiclass ROC Curve (OvR with Micro-Average)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.show()

    for i in range(6):
        prob_true, prob_pred = calibration_curve(
            all_labels[:, i], all_preds[:, i], n_bins=10
        )
        width = 0.007
        offset = (i - 6/2) * width * 1.5
        #offset = 0
        #plt.bar(prob_pred + offset, prob_true, width=width, alpha=0.7, label=f"Class {i}")
        plt.plot(prob_pred, prob_true, label=label_cols[i])

    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
        #plt.show()
    plt.title(kernel + " Multiclass Probability Calibration Curves (OvR)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid(True)
    plt.show()

    # find optimal temperature
    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def nll(logits, labels, T):
        scaled = logits / T
        probs = softmax(scaled)
        return -np.mean(np.log(probs[np.arange(len(labels)), labels]))

    def find_temperature(logits, labels):
        def objective(logT):
            T = np.exp(logT)
            return nll(logits, labels, T)
        res = minimize(objective, x0=[0.0], method='L-BFGS-B')
        return float(np.exp(res.x[0]))

    logits = all_preds.detach().cpu().numpy()
    labels = all_labels.detach().cpu().numpy().astype(int)
    labels_int = labels.argmax(axis=1)  
    T = find_temperature(logits, labels_int)
    print("Optimal T:", T)
    all_labels = all_labels.cpu().numpy()
    temp_preds = softmax(logits / T)
    for i in range(6):
        #y_true = (all_labels == i).astype(int)
        prob_true, prob_pred = calibration_curve(
            all_labels[:, i], temp_preds[:, i], n_bins=10
        )
        width = 0.007
        offset = (i - 6/2) * width * 1.5
        #offset = 0
        #plt.bar(prob_pred + offset, prob_true, width=width, alpha=0.7, label=f"Class {i}")
        plt.plot(prob_pred, prob_true, label=label_cols[i])

        plt.plot([0, 1], [0, 1], "k--", linewidth=2)
        #plt.show()
    plt.title(kernel + " Multiclass Probability Calibration Curves with temperature scaling")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid(True)
    plt.show()

    def expected_calibration_error(probs, labels, n_bins=15):
        """ECE: weighted average of |accuracy - confidence| across bins"""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i+1]
            mask = (confidences > low) & (confidences <= high) if i>0 else (confidences >= low) & (confidences <= high)
            if np.sum(mask) == 0:
                continue
            acc = np.mean(predictions[mask] == labels[mask])
            conf = np.mean(confidences[mask])
            ece += np.sum(mask) / len(labels) * abs(acc - conf)
            mce = max(mce, abs(acc - conf))
        return ece, mce

    ece, mce = expected_calibration_error(temp_preds, labels_int)
    brier = brier_score_loss(np.eye(6)[labels_int], temp_preds)
    logloss = log_loss(labels_int, temp_preds)

    print(f"{kernel} ECE: {ece:.4f}") # Expected calibration error -> really good
    print(f"{kernel}  MCE: {mce:.4f}") # Mean calibration error -> some are really far off
    print(f"{kernel}  Brier score: {brier:.4f}") # brier score is okay
    print(f"{kernel}  Log-loss: {logloss:.4f}") # log loss is really good


    pca = PCA()
    print("Variance explained: ",pca.fit(all_emb).explained_variance_ratio_)

    all_emb_norm = all_emb / np.linalg.norm(all_emb, axis = 1, keepdims= True)
    all_emb_norm_mean_centered = all_emb_norm - all_emb_norm.mean(axis = 0, keepdims = True)
    all_emb_norm_mean_centered_pca = PCA().fit_transform(all_emb_norm_mean_centered)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(all_emb_norm_mean_centered_pca[:,0], all_emb_norm_mean_centered_pca[:,1], all_emb_norm_mean_centered_pca[:,2], c = labels_int, cmap = "tab10" )
    handles = []
    colors = [scatter.cmap(scatter.norm(i)) for i in range(len(label_cols))]
    for color, label in zip(colors, label_cols):
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles, loc='best')
    ax.set_title(kernel + " Mean centered PCA")
    plt.show()

def analyze_dataset(df, sequence_col="Sequence"):
    print("===== DATASET SUMMARY =====")
    # --- Basic shape ---
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print()

    # --- Sequence lengths ---
    seq_lengths = df[sequence_col].apply(len)

    print("===== SEQUENCE LENGTH STATISTICS =====")
    print(f"Min length:   {seq_lengths.min()}")
    print(f"Max length:   {seq_lengths.max()}")
    print(f"Mean length:  {seq_lengths.mean():.2f}")
    print(f"Median length:{seq_lengths.median():.2f}")
    print()

    print("===== LABEL DISTRIBUTION =====")
    
    label_counts = df["class"].value_counts().sort_values(ascending=False)
    
    for label, count in label_counts.items():
        print(f"{label:25s}: {int(count)}")
    print()

    print("===== DONE =====")