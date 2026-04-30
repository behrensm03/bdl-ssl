import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

def plot_loss_curves_bcnn(history, unlabeled_rate=0, use_unlabeled=False):
    epochs = [h['epoch'] for h in history]
    train_nll = [h['train_nll'] for h in history]
    train_kl = [h['train_kl_avg'] for h in history]
    val_nll = [h['val_nll'] for h in history]
    
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_nll, label='Train NLL (Labeled)')
    plt.plot(epochs, val_nll, label='Val NLL')
    if use_unlabeled:
        train_loss_unlabeled = [h['train_loss_unlabeled'] for h in history]
        plt.plot(epochs, train_loss_unlabeled, label='Train Loss (Unlabeled)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'BCNN Training Loss Curves (Unlabeled Rate: {unlabeled_rate})')
    plt.legend()
    plt.show()

def plot_kl(history):
    epochs = [h['epoch'] for h in history]
    train_kl = [h['train_kl_avg'] for h in history]
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_kl, label='Train KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('Average KL Divergence Over Time')
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

def aggregate_seed_results(results_by_seed):
    seeds = list(results_by_seed.keys())
    keys = results_by_seed[seeds[0]].keys()
    aggregated = {}
    for key in keys:
        values = [results_by_seed[seed][key] for seed in seeds]
        aggregated[f"{key}_mean"] = np.mean(values, axis=0)
        aggregated[f"{key}_std"] = np.std(values, axis=0)
    return aggregated

def print_aggregate_test_results(results_by_seed):
    avg_test_results = aggregate_seed_results(results_by_seed)
    print(f"Test mAUC: {avg_test_results['macro_auc_mean']:.4f} ± {avg_test_results['macro_auc_std']:.4f}")
    print(f"Test mNLL: {avg_test_results['macro_nll_mean']:.4f} ± {avg_test_results['macro_nll_std']:.4f}")
    per_class_mean = avg_test_results['per_class_nll_mean']
    per_class_std = avg_test_results['per_class_nll_std']
    per_class_str = '  '.join([f"{m:.4f} ± {s:.4f}" for m, s in zip(per_class_mean, per_class_std)])
    print(f"Test per-class NLL: [{per_class_str}]")
    per_class_auc_mean = avg_test_results['per_class_auc_mean']
    per_class_auc_std = avg_test_results['per_class_auc_std']
    per_class_auc_str = '  '.join([f"{m:.4f} ± {s:.4f}" for m, s in zip(per_class_auc_mean, per_class_auc_std)])
    print(f"Test per-class AUC: [{per_class_auc_str}]")

