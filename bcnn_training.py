import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# TODO: implement the device moving cpu/gpu

def elbo_loss(model, outputs, targets, criterion, beta=1.0):
    # criterion is nn.crossentropyloss
    nll = criterion(outputs, targets)
    # divide kl bythe number of batches and add it to NLL to compute negative elbo
    num_params = sum(p.numel() for p in model.parameters())
    kl = beta * model.kl_divergence() / num_params
    loss = nll + kl
    return loss, nll.detach(), kl.detach()

def train_loop_bcnn_hard_pseudo_label(model, train_loader, val_loader, criterion, optimizer, num_epochs, threshold=0.95, alpha=0.5, beta=1.0, num_samples=10, n_classes=7, device='cpu'):
    # At each epoch, we need to compute the ELBO
    # First, we need to figure out the set U of onehot vectors from pseudo labels
    # So, run the unlabeled examples through the network S times, and then get the average probability vector for each unlabeled example
    # From there, using those average prob vectors, figure out which meet the threshold and that defines U.
    # Next, we need to compute the ELBO value. Start by running the labeled examples through the network once, and compute the CE loss using the ground truth labels
    # Then, run the examples in U through the network once, and compute the CE loss using those pseudo labels in U.
    # Next, compute the KL divergence for the network and then add those three terms together (weighting the unlabeled loss by alpha).
    # We may need to scale the KL divergence term if it dominates.

    history = []
    for epoch in range(num_epochs):
        model.train()
        train_nll, train_loss_unlabeled, train_kl = 0.0, 0.0, 0.0
        n_labeled, n_unlabeled, n_unlabeled_seen = 0, 0, 0
        # First, the training loop
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # separate the labeled and unlabeled examples
            label_mask = (labels != -1).squeeze()
            unlabeled_mask = (labels == -1).squeeze()
            inputs_labeled, inputs_unlabeled = images[label_mask], images[unlabeled_mask]
            targets_labeled = labels[label_mask]

            losses = []
            if inputs_labeled.size(0) > 0:
                labeled_outputs = model(inputs_labeled)
                targets_labeled = targets_labeled.squeeze().long()
                loss, nll, kl = elbo_loss(model, labeled_outputs, targets_labeled, criterion, beta=beta)
                losses.append(loss)
                train_nll += nll.item() * inputs_labeled.size(0)
                train_kl += kl.item()
                n_labeled += inputs_labeled.size(0)
            
            if inputs_unlabeled.size(0) != 0:
                n_unlabeled_seen += inputs_unlabeled.size(0)
                # First get average prob vector
                with torch.no_grad():
                    # should this be in eval mode?
                    avg_probs = model.average_probs(inputs_unlabeled, num_samples=num_samples)
                # compute pseudo labels using avg probs
                max_probs, pseudo_labels = torch.max(avg_probs, dim=1)
                confident_mask = max_probs >= threshold
                if confident_mask.sum() > 0:
                    # rerun the examples in U through the network once and softmax
                    unlabeled_inputs_keep = inputs_unlabeled[confident_mask]
                    unlabeled_outputs_keep = model(unlabeled_inputs_keep)
                    pseudo_labels_keep = pseudo_labels[confident_mask]
                    loss_unlabeled = criterion(unlabeled_outputs_keep, pseudo_labels_keep)
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
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                targets = labels.squeeze().long().to(device)

                # Monte Carlo predictive average
                mc_probs = []
                for _ in range(num_samples):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    mc_probs.append(probs)

                mean_probs = torch.stack(mc_probs, dim=0).mean(dim=0)

                val_total += images.size(0)

                nll_val = -torch.log(mean_probs[torch.arange(len(targets)), targets] + 1e-8).mean()
                val_nll += nll_val.item() * images.size(0)

                # store predictions and targets for auc
                val_probs.append(mean_probs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_probs_np = np.concatenate(val_probs)
        val_targets_np = np.concatenate(val_targets)

        # per class NLL on val
        per_class_nll = []
        for i in range(n_classes):
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
            "train_kl_avg": train_kl / len(train_loader) if len(train_loader) > 0 else 0.0,
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "n_unlabeled_seen": n_unlabeled_seen,
            "val_nll": val_nll / val_total if val_total > 0 else 0.0,
            "val_macro_nll": val_macro_nll,
            "val_per_class_nll": per_class_nll,
            "val_auc_macro": roc_auc_score(val_targets_np, val_probs_np, multi_class='ovr', average='macro'),
            "val_auc_global": roc_auc_score(val_targets_np, val_probs_np, multi_class='ovr', average='micro'),
            "model_state": {k: v.clone() for k,v in model.state_dict().items()},
        }

        history.append(summary)
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train NLL: {train_nll/n_labeled if n_labeled > 0 else 0.0:.4f} | "
            f"Train KL (avg/batch): {train_kl/len(train_loader):.4f} | "
            f"Train Loss Unlabeled: {train_loss_unlabeled/n_unlabeled if n_unlabeled > 0 else 0.0:.4f} | "
            f"Unlabeled Examples Used: {n_unlabeled}/{n_unlabeled_seen} | "
            f"Val NLL: {summary['val_nll']:.4f} | "
            f"Val Macro NLL: {summary['val_macro_nll']:.4f} | "
            f"Val AUC Macro: {summary['val_auc_macro']:.4f} | "
            f"Val AUC Global: {summary['val_auc_global']:.4f}")
        
    return history

def train_loop_bcnn_soft_pseudo_label(model, train_loader, val_loader, criterion, optimizer, num_epochs, alpha=0.5, beta=1.0, num_samples=10):
    # similar to hard pl loop but using the average prob vector as the pseudo label
    history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss_labeled, train_loss_unlabeled, train_total_labeled, train_total_unlabeled, train_total_unlabeled_seen, train_kl_total = 0.0, 0.0, 0, 0, 0, 0.0

        for images, labels in tqdm(train_loader):
            label_mask = (labels != -1).squeeze()
            unlabeled_mask = (labels == -1).squeeze()
            inputs_labeled, inputs_unlabeled = images[label_mask], images[unlabeled_mask]
            targets_labeled = labels[label_mask]

            losses = []
            if inputs_labeled.size(0) > 0:
                labeled_outputs = model(inputs_labeled)
                targets_labeled = targets_labeled.squeeze().long()
                loss, nll, kl = elbo_loss(model, labeled_outputs, targets_labeled, criterion, beta=beta)
                losses.append(loss)
                train_loss_labeled += nll.item() * inputs_labeled.size(0)
                train_kl_total += kl.item()
                train_total_labeled += inputs_labeled.size(0)

            if inputs_unlabeled.size(0) != 0:
                train_total_unlabeled_seen += inputs_unlabeled.size(0)
                with torch.no_grad():
                    # should this be in eval mode?
                    pseudo_labels = model.average_probs(inputs_unlabeled, num_samples=num_samples)
                
                # avg probs are the pseudo labels now
                unlabeled_outputs = model(inputs_unlabeled)
                # apparently nn.CrossEntropyLoss won't take soft pseudo labels so we directly use F.cross_entropy
                loss_unlabeled = F.cross_entropy(unlabeled_outputs, pseudo_labels)
                losses.append(alpha * loss_unlabeled)
                train_loss_unlabeled += loss_unlabeled.item() * inputs_unlabeled.size(0)
                train_total_unlabeled += inputs_unlabeled.size(0)

            if len(losses) > 0:
                optimizer.zero_grad()
                sum(losses).backward()
                optimizer.step()

        model.eval()
        val_loss, val_total = 0.0, 0
        val_probs, val_targets = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                targets = labels.squeeze().long()

                mc_probs = []
                for _ in range(num_samples):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    mc_probs.append(probs)

                mean_probs = torch.stack(mc_probs, dim=0).mean(dim=0)

                val_total += images.size(0)

                nll_val = -torch.log(mean_probs[torch.arange(len(targets)), targets] + 1e-8).mean()
                val_loss += nll_val.item() * images.size(0)

                val_probs.append(mean_probs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        summary = {
            "epoch": epoch+1,
            "train_nll_labeled": train_loss_labeled / train_total_labeled if train_total_labeled > 0 else 0.0,
            "train_loss_unlabeled": train_loss_unlabeled / train_total_unlabeled if train_total_unlabeled > 0 else 0.0,
            "val_loss": val_loss / val_total if val_total > 0 else 0.0,
            "val_auc_macro": roc_auc_score(np.concatenate(val_targets), np.concatenate(val_probs), multi_class='ovr', average='macro'),
            "val_auc_global": roc_auc_score(np.concatenate(val_targets), np.concatenate(val_probs), multi_class='ovr', average='micro'),
            "train_total_labeled": train_total_labeled,
            "train_total_unlabeled": train_total_unlabeled,
            "train_total_unlabeled_seen": train_total_unlabeled_seen,
            "val_total": val_total,
            "model_state": {k: v.clone() for k,v in model.state_dict().items()},
            "train_kl_total": train_kl_total,
            "train_kl_avg": train_kl_total / len(train_loader) if len(train_loader) > 0 else 0.0,
        }

        history.append(summary)
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train NLL: {train_loss_labeled/train_total_labeled if train_total_labeled > 0 else 0.0:.4f} | "
            f"Train KL (avg/batch): {train_kl_total/len(train_loader):.4f} | "
            f"Unlabeled Examples Used: {train_total_unlabeled}/{train_total_unlabeled_seen} | "
            f"Val Loss: {summary['val_loss']:.4f} | "
            f"Train Loss Unlabeled: {train_loss_unlabeled/train_total_unlabeled if train_total_unlabeled > 0 else 0.0:.4f} | "
            f"Val AUC Macro: {summary['val_auc_macro']:.4f} | "
            f"Val AUC Global: {summary['val_auc_global']:.4f}")
        
    return history

@torch.no_grad()
def evaluate_bayesian(model, test_loader, device, mc_samples=20, n_classes=7):
    model.eval()
    probs, targets = [], []

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        mean_probs = model.average_probs(images, num_samples=mc_samples)
        probs.append(mean_probs.cpu().numpy())
        targets.append(labels.squeeze().cpu().numpy())

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    macro_auc = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
    global_auc = roc_auc_score(targets, probs, multi_class="ovr", average="micro")
    nll = -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-10))  # add small value for numerical stability

    targets_binarized = label_binarize(targets, classes=np.arange(n_classes))
    per_class_auc = [roc_auc_score(targets_binarized[: , i], probs[:, i]) for i in range(n_classes)]

    preds = np.argmax(probs, axis=1)
    matrix = confusion_matrix(targets, preds, normalize='true')

    # Per-class NLL to see if the model is taking advantage of class imbalance
    per_class_nll = []
    for i in range(n_classes):
        mask = (targets == i)
        if mask.sum() > 0:
            class_nll = -np.mean(np.log(probs[mask, i] + 1e-10))
            per_class_nll.append(class_nll)
        else:
            per_class_nll.append(None)
    macro_nll = np.mean([nll for nll in per_class_nll if nll is not None])

    return {
        "macro_auc": macro_auc,
        "global_auc": global_auc,
        "nll": nll,
        "per_class_auc": per_class_auc,
        "per_class_nll": np.array(per_class_nll),
        "macro_nll": macro_nll,
        "confusion_matrix": matrix
    }


