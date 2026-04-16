import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def train_loop_bcnn_hard_pseudo_label(model, train_loader, val_loader, criterion, optimizer, num_epochs, threshold=0.95, alpha=0.5, num_samples=10):
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
        train_loss_labeled, train_loss_unlabeled, train_total_labeled, train_total_unlabeled, train_total_unlabeled_seen = 0.0, 0.0, 0, 0, 0
        # First, the training loop
        for images, labels in tqdm(train_loader):
            # separate the labeled and unlabeled examples
            label_mask = labels != -1
            unlabeled_mask = labels == -1
            inputs_labeled, inputs_unlabeled = images[label_mask], images[unlabeled_mask]
            targets_labeled = labels[label_mask]

            losses = []
            if inputs_labeled.size(0) > 0:
                labeled_outputs = model(inputs_labeled)
                targets_labeled = targets_labeled.squeeze().long()
                loss_labeled = criterion(labeled_outputs, targets_labeled)
                losses.append(loss_labeled)
                train_loss_labeled += loss_labeled.item() * inputs_labeled.size(0)
                train_total_labeled += inputs_labeled.size(0)
            
            if inputs_unlabeled.size(0) != 0:
                train_total_unlabeled_seen += inputs_unlabeled.size(0)
                # First get average prob vector
                avg_probs = model.average_probs(inputs_unlabeled, num_samples=num_samples)
                # compute pseudo labels using avg probs
                max_probs, pseudo_labels = torch.max(avg_probs, dim=1)
                confident_mask = max_probs >= threshold
                if confident_mask.sum() > 0:
                    # rerun the examples in U through the network once and softmax
                    unlabeled_inputs_keep = inputs_unlabeled[confident_mask]
                    unlabeled_outputs_keep = torch.softmax(model(unlabeled_inputs_keep), dim=1)
                    pseudo_labels_keep = pseudo_labels[confident_mask]
                    loss_unlabeled = criterion(unlabeled_outputs_keep, pseudo_labels_keep)
                    losses.append(alpha * loss_unlabeled)
                    train_loss_unlabeled += loss_unlabeled.item() * unlabeled_outputs_keep.size(0)
                    train_total_unlabeled += unlabeled_outputs_keep.size(0)

            if len(losses) > 0:
                optimizer.zero_grad()
                sum(losses).backward()
                optimizer.step()

        # Then the validation loop
