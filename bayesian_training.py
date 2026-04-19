import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def elbo_loss(model, outputs, targets, criterion, num_batches):
    #criterion is nn.crossentropyloss
    nll = criterion(outputs, targets)
    kl = model.kl_divergence()
    #divide kl bythe number of batches and add it to NLL to compute negative 
    #elbo
    loss = nll + kl / num_batches
    return loss, nll.detach(), kl.detach()


def train_loop_bayesian_labeled(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):

    history = []
    num_batches = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        train_nll_total = 0.0
        train_kl_total = 0.0
        train_total = 0

        for images, labels in tqdm(train_loader):
            label_mask = (labels != -1).squeeze()
            if label_mask.sum() == 0:
                continue
                
            #moving to cpu/gpu
            images = images[label_mask].to(device)
            targets = labels[label_mask].squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, nll, kl = elbo_loss(model, outputs, targets, criterion, num_batches)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            train_loss_total += loss.item() * bs
            train_nll_total += nll.item() * bs
            train_kl_total += kl.item() * bs
            train_total += bs


        # Then the validation loop
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                targets = labels.squeeze().long().to(device)

                #Monte Carlo predictive average
                mc_probs = []
                for _ in range(10):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    mc_probs.append(probs)

                mean_probs = torch.stack(mc_probs, dim=0).mean(dim=0)
                preds = mean_probs.argmax(dim=1)

                val_loss += batch_loss.item() * images.size(0)
                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()

                # store predictions and targets for auc
                val_probs.append(mean_probs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss_total / train_total,
            "train_nll": train_nll_total / train_total,
            "train_kl": train_kl_total / train_total,
            "val_loss": val_loss / val_total,
            "val_acc": val_correct / val_total,
            "val_auc": roc_auc_score(
                np.concatenate(val_targets),
                np.concatenate(val_probs),
                multi_class="ovr",
                average="macro",
            ),
            "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        }

        history.append(summary)
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {summary['train_loss']:.4f} | "
            f"Train NLL: {summary['train_nll']:.4f} | "
            f"Train KL: {summary['train_kl']:.4f} | "
            f"Val Loss: {summary['val_loss']:.4f} | "
            f"Val Acc: {summary['val_acc']:.4f} | "
            f"Val AUC: {summary['val_auc']:.4f}"
        )

    return history

# @torch.no_grad should make runtime faster
@torch.no_grad()
def evaluate_bayesian(model, test_loader, device, mc_samples=20):
    model.eval()
    probs = []
    targets = []

    #loop overtest data
    for images, labels in test_loader:
        #for moving to cpu/gpu
        images = images.to(device)
        mc_probs = []

        # monte ccarlo sampling loop
        for _ in range(mc_samples):
            outputs = model(images)
            mc_probs.append(torch.softmax(outputs, dim=1))

        #average the predictions
        mean_probs = torch.stack(mc_probs, dim=0).mean(dim=0)
    
        probs.append(mean_probs.cpu().numpy())
        targets.append(labels.squeeze().cpu().numpy())

    #combine batches
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    #compute auc and accuracy
    #the auc function is apparently not multiclass so added argument for it
    #macro will average the auc into one score for all classes. or do we want
    # one auc for each class? 
    auc = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
    acc = (probs.argmax(axis=1) == targets).mean()

    #add confusion matrix
    #TODO

    return {"test_auc": auc, "test_acc": acc}