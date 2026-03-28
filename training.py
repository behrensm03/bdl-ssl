import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

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
    return semi_supervised_labels

def train_loop_labeled(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # This is the training loop utilizing only labeled examples
    history = []
    for epoch in range(num_epochs):
        # train_correct, train_total, test_correct, test_total = 0, 0, 0, 0
        model.train()
        train_loss, train_total = 0.0, 0
        for images, labels in tqdm(train_loader):
            label_mask = labels != -1
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
            train_loss += loss.item() * inputs.size(0) # need to multiply by batch size because loss is averaged over the batch, and batches are different sizes because of unlabeled examples
            train_total += inputs.size(0)
        
        # evaluate on validation set
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                targets = labels.squeeze().long()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        summary = {
            "epoch": epoch+1,
            "train_loss": train_loss/train_total,
            "val_loss": val_loss/val_total,
            "val_acc": val_correct/val_total,
            'train_total': train_total,
            'val_total': val_total
        }
        history.append(summary)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} | "
              f"Val Acc: {val_correct/val_total:.4f}")
        
    return history


def train_loop_unlabeled(model, train_loader, val_loader, criterion, optimizer, num_epochs, threshold=0.95, alpha=0.5):
    # This is the training loop utilizing all examples, with standard soft pseudo-labeling and threshold
    history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss_labeled, train_loss_unlabeled, train_total_labeled, train_total_unlabeled = 0.0, 0.0, 0, 0
        # First the training loop
        for images, labels in tqdm(train_loader):
            # separate the labeled and unlabeled examples
            label_mask = labels != -1
            unlabeled_mask = labels == -1
            inputs_labeled, inputs_unlabeled = images[label_mask], images[unlabeled_mask]
            targets_labeled = labels[label_mask]

            loss = torch.tensor(0.0, requires_grad=True)
            if inputs_labeled.size(0) != 0:
                labeled_outputs = model(inputs_labeled)
                targets_labeled = targets_labeled.squeeze().long()
                loss_labeled = criterion(labeled_outputs, targets_labeled)
                loss = loss + loss_labeled
                train_loss_labeled += loss_labeled.item() * inputs_labeled.size(0)
                train_total_labeled += inputs_labeled.size(0)

            if inputs_unlabeled.size(0) != 0:
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
                    loss = loss + alpha * loss_unlabeled
                    train_loss_unlabeled += loss_unlabeled.item() * unlabeled_outputs_keep.size(0)
                    train_total_unlabeled += unlabeled_outputs_keep.size(0)

            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Then the validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                # We only have labeled examples in validation and test data
                outputs = model(images)
                targets = labels.squeeze().long()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, predictions = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predictions == targets).sum().item()

        summary = {
            "epoch": epoch+1,
            "train_loss_labeled": train_loss_labeled/train_total_labeled if train_total_labeled > 0 else 0.0,
            "train_loss_unlabeled": train_loss_unlabeled/train_total_unlabeled if train_total_unlabeled > 0 else 0.0,
            "val_loss": val_loss/val_total,
            "val_acc": val_correct/val_total,
            'train_total_labeled': train_total_labeled,
            'train_total_unlabeled': train_total_unlabeled,
            'val_total': val_total
        }

        history.append(summary)
        print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss Labeled: {train_loss_labeled/train_total_labeled if train_total_labeled > 0 else 0.0:.4f} | "
                f"Train Loss Unlabeled: {train_loss_unlabeled/train_total_unlabeled if train_total_unlabeled > 0 else 0.0:.4f} | "
                f"Val Loss: {val_loss/val_total:.4f} | "
                f"Val Acc: {val_correct/val_total:.4f}")

