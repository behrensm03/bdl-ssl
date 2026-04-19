import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.25)

def plot_loss_curves(history, unlabeled_rate=0, use_unlabeled=False):
    # History is a list of dictionaries, where each dictionary contains the metrics for one epoch
    epochs = [h['epoch'] for h in history]
    train_loss_labeled = [h['train_loss_labeled'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_loss_labeled, label='Train Loss (Labeled)')
    if use_unlabeled:
        train_loss_unlabeled = [h['train_loss_unlabeled'] for h in history]
        plt.plot(epochs, train_loss_unlabeled, label='Train Loss (Unlabeled)')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curves (Unlabeled Rate: {unlabeled_rate})')
    plt.legend()
    plt.show()

def plot_auc_curve(history, unlabeled_rate=0):
    epochs = [h['epoch'] for h in history]
    val_auc_macro = [h['val_auc_macro'] for h in history]
    plt.figure(figsize=(6,4))
    plt.plot(epochs, val_auc_macro, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC (macro OvR)')
    plt.title(f'Validation AUC Curve (Unlabeled Rate: {unlabeled_rate})')
    plt.legend()
    plt.show()

def plot_perclass_auc(perclass_auc, class_names):
    plt.figure(figsize=(6,4))
    plt.bar(class_names, perclass_auc)
    for i, v in enumerate(perclass_auc):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('Class')
    plt.ylabel('AUC (OvR)')
    plt.title('Per-Class AUC')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_per_class_recall(confusion_matrix, class_names):
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix (Recall)')
    plt.show()

def plot_percent_unlabeled_used(history):
    epochs = [h['epoch'] for h in history]
    percent_unlabeled_used = [h['train_total_unlabeled']/h['train_total_unlabeled_seen']*100 if h['train_total_unlabeled_seen'] > 0 else 0 for h in history]
    plt.figure(figsize=(6,4))
    plt.plot(epochs, percent_unlabeled_used, label='Percent Unlabeled Used')
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.title('Percent of Unlabeled Examples Used Over Time')
    plt.show()