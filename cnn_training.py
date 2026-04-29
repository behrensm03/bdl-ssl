import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# we can implement our own version of the torch Dataset that will handle our unlabeled examples
class SSLDataset(data.Dataset):
    def __init__(self, dataset, ssl_labels):
        self.dataset = dataset
        self.ssl_labels = ssl_labels

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx] # throw away the original label
        ssl_label = self.ssl_labels[idx]
        return img, ssl_label
    

def get_semi_supervised_labels(dataset, unlabeled_rate, seed=42, n_classes=7):
    labels = np.array(dataset.labels).flatten().astype(np.int16)
    semi_supervised_labels = labels.copy()
    np.random.seed(seed)
    for c in range(n_classes):
        # find the indices of the full examples that belong to class c
        class_indices = np.where(labels == c)[0]
        # next, randomly choose a subset of those indices to be unlabeled, according to the specified unlabeled rate
        num_unlabeled = int(len(class_indices) * unlabeled_rate)
        unlabeled_indices = np.random.choice(class_indices, size=num_unlabeled, replace=False)
        # set the labels of those indices to -1
        semi_supervised_labels[unlabeled_indices] = -1

    print(f'Unlabeled rate: {unlabeled_rate} | Total examples: {len(labels)} | Labeled examples: {(semi_supervised_labels != -1).sum()} | Unlabeled examples: {(semi_supervised_labels == -1).sum()}')
    for c in range(n_classes):
        total = (labels == c).sum()
        labeled = (semi_supervised_labels == c).sum()
        unlabeled = total - labeled
        print(f'Class {c}: {labeled}/{total} labeled, {unlabeled} unlabeled')
    return semi_supervised_labels

def train_loop_labeled(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_classes=7, device='cpu'):
    # This is the training loop utilizing only labeled examples
    history = []
    for epoch in range(num_epochs):
        # train_correct, train_total, test_correct, test_total = 0, 0, 0, 0
        model.train()
        train_nll, n_labeled = 0.0, 0
        for images, labels in tqdm(train_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            label_mask = (labels != -1).squeeze()
            if label_mask.sum() == 0:
                continue # skip batches with no labeled examples

            inputs = images[label_mask]
            targets = labels[label_mask]

            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_nll += loss.item() * inputs.size(0) # need to multiply by batch size because loss is averaged over the batch, and batches are different sizes because of unlabeled examples
            n_labeled += inputs.size(0)
        
        # evaluate on validation set
        model.eval()
        val_nll, val_total = 0.0, 0
        val_probs, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, targets = images.to(device), labels.to(device).squeeze().long()
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_nll += loss.item() * images.size(0)

                val_total += targets.size(0)

                val_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_probs_np = np.concatenate(val_probs)
        val_targets_np = np.concatenate(val_targets)

        per_class_nll = []
        for i in range(num_classes):
            mask = (val_targets_np == i)
            if mask.sum() > 0:
                class_nll = -np.mean(np.log(val_probs_np[mask, i] + 1e-8))
                per_class_nll.append(class_nll)
            else:
                per_class_nll.append(None)
        val_macro_nll = np.mean([nll for nll in per_class_nll if nll is not None])

        summary = {
            "epoch": epoch+1,
            "train_nll": train_nll / n_labeled if n_labeled > 0 else 0.0,
            "n_labeled": n_labeled,
            "val_nll": val_nll / val_total if val_total > 0 else 0.0,
            "val_macro_nll": val_macro_nll,
            "val_per_class_nll": per_class_nll,
            "val_auc_macro": roc_auc_score(val_targets_np, val_probs_np, multi_class='ovr', average='macro'),
            'model_state': {k: v.clone() for k,v in model.state_dict().items()}
        }
        history.append(summary)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train NLL: {summary['train_nll']:.4f} | "
              f"Val NLL: {summary['val_nll']:.4f} | "
              f"Val Macro NLL: {summary['val_macro_nll']:.4f} | "
              f"Val mAUC: {summary['val_auc_macro']:.4f}")
        
    return history

def train_loop_hard_pseudo_label(model, train_loader, val_loader, criterion, optimizer, num_epochs, threshold=0.95, alpha=0.5, num_classes=7, device='cpu'):
    # This is the training loop utilizing all examples, with threshold-based one-hot pseudo labeling for the unlabeled examples
    history = []
    for epoch in range(num_epochs):
        model.train()
        train_nll, train_loss_unlabeled = 0.0, 0.0
        n_labeled, n_unlabeled, n_unlabeled_seen = 0, 0, 0
        # First the training loop
        for images, labels in tqdm(train_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            # separate the labeled and unlabeled examples
            label_mask = (labels != -1).squeeze()
            unlabeled_mask = (labels == -1).squeeze()
            inputs_labeled, inputs_unlabeled = images[label_mask], images[unlabeled_mask]
            targets_labeled = labels[label_mask]

            losses = []
            if inputs_labeled.size(0) != 0:
                labeled_outputs = model(inputs_labeled)
                targets_labeled = targets_labeled.squeeze().long()
                loss_labeled = criterion(labeled_outputs, targets_labeled)
                losses.append(loss_labeled)
                train_nll += loss_labeled.item() * inputs_labeled.size(0)
                n_labeled += inputs_labeled.size(0)

            if inputs_unlabeled.size(0) != 0:
                n_unlabeled_seen += inputs_unlabeled.size(0)
                unlabeled_outputs = model(inputs_unlabeled)
                # now use the threshold to generate pseudo-labels from the unlabeled outputs
                probs_unlabeled = torch.softmax(unlabeled_outputs, dim=1)
                max_probs, pseudo_labels = torch.max(probs_unlabeled, dim=1)
                # use a mask to keep only the confident probabilities
                confident_mask = max_probs >= threshold
                # if there are any confident examples, compute the loss on those as well
                if confident_mask.sum() > 0:
                    unlabeled_outputs_keep = unlabeled_outputs[confident_mask]
                    pseudo_labels_keep = pseudo_labels[confident_mask]
                    loss_unlabeled = criterion(unlabeled_outputs_keep, pseudo_labels_keep)
                    # combine the losses in a weighted manner
                    losses.append(alpha * loss_unlabeled)
                    train_loss_unlabeled += loss_unlabeled.item() * unlabeled_outputs_keep.size(0)
                    n_unlabeled += unlabeled_outputs_keep.size(0)

            if len(losses) > 0:
                optimizer.zero_grad()
                sum(losses).backward()
                optimizer.step()

        # Then the validation loop
        model.eval()
        val_nll, val_total = 0.0, 0
        val_probs, val_targets = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, leave=False):
                # We only have labeled examples in validation and test data
                images, targets = images.to(device), labels.to(device).squeeze().long()
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_nll += loss.item() * images.size(0)
                val_total += targets.size(0)

                # store predictions and targets for auc
                val_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_probs_np = np.concatenate(val_probs)
        val_targets_np = np.concatenate(val_targets)

        per_class_nll = []
        for i in range(num_classes):
            mask = (val_targets_np == i)
            if mask.sum() > 0:
                class_nll = -np.mean(np.log(val_probs_np[mask, i] + 1e-8))
                per_class_nll.append(class_nll)
            else:
                per_class_nll.append(None)
        val_macro_nll = np.mean([nll for nll in per_class_nll if nll is not None])

        summary = {
            "epoch": epoch+1,
            "train_nll": train_nll / n_labeled if n_labeled > 0 else 0.0,
            "train_loss_unlabeled": train_loss_unlabeled / n_unlabeled if n_unlabeled > 0 else 0.0,
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "n_unlabeled_seen": n_unlabeled_seen,
            "val_nll": val_nll / val_total if val_total > 0 else 0.0,
            "val_macro_nll": val_macro_nll,
            "val_per_class_nll": per_class_nll,
            "val_auc_macro": roc_auc_score(val_targets_np, val_probs_np, multi_class='ovr', average='macro'),
            'model_state': {k: v.clone() for k,v in model.state_dict().items()},
        }

        history.append(summary)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"NLL: {summary['train_nll']:.4f} | "
              f"Unlabeled Loss: {summary['train_loss_unlabeled']:.4f} | "
              f"Unlabeled: {n_unlabeled}/{n_unlabeled_seen} | "
              f"Val NLL: {summary['val_nll']:.4f} | "
              f"Val Macro NLL: {summary['val_macro_nll']:.4f} | "
              f"Val mAUC: {summary['val_auc_macro']:.4f}")

    return history

def evaluate(model, test_loader, n_classes=7, device='cpu'):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            targets.append(labels.squeeze().cpu().numpy())

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    per_class_nll = []
    for i in range(n_classes):
        mask = (targets == i)
        if mask.sum() > 0:
            class_nll = -np.mean(np.log(probs[mask, i] + 1e-10))
            per_class_nll.append(class_nll)
        else:
            per_class_nll.append(None)

    macro_nll = np.mean([nll for nll in per_class_nll if nll is not None])

    preds = np.argmax(probs, axis=1)
    matrix = confusion_matrix(targets, preds, normalize='true')

    targets_binarized = label_binarize(targets, classes=np.arange(n_classes))
    per_class_auc = [roc_auc_score(targets_binarized[:, i], probs[:, i]) for i in range(n_classes)]

    nll = -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-10))  # add small value for numerical stability

    return {
        "macro_auc": roc_auc_score(targets, probs, multi_class='ovr', average='macro'),
        "nll": nll,
        "macro_nll": macro_nll,
        "per_class_nll": np.array(per_class_nll),
        "per_class_auc": np.array(per_class_auc),
        "confusion_matrix": matrix
    }

