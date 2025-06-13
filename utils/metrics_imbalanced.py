from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_imbalanced_metrics(y_true, y_pred_proba, threshold=None):
    """
    Calculate comprehensive metrics for imbalanced classification.
    
    Args:
        y_true: True labels (numpy array)
        y_pred_proba: Predicted probabilities (numpy array) - can be 2D or 1D
        threshold: Optional fixed threshold; if None, finds optimal
    
    Returns:
        Dictionary containing all relevant metrics
    """
    # Handle different probability formats
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
        # Multi-class probabilities - extract positive class
        pos_proba = y_pred_proba[:, 1]
    else:
        pos_proba = y_pred_proba.squeeze()
    
    # Core metrics that don't require threshold
    auc_roc = roc_auc_score(y_true, pos_proba)
    auc_pr = average_precision_score(y_true, pos_proba)
    
    # Find optimal threshold if not provided
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_true, pos_proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Find threshold that maximizes F1
        optimal_idx = np.nanargmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    else:
        optimal_threshold = threshold
    
    # Calculate predictions at optimal threshold
    y_pred = (pos_proba >= optimal_threshold).astype(int)
    
    # Calculate detailed metrics
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate per-class accuracy
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    return metrics

def plot_imbalanced_metrics(metrics, save_path=None):
    """
    Create visualization plots for imbalanced classification metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_imbalanced_metrics
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Metrics Bar Plot
    metric_names = ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [
        metrics['auc_roc'], metrics['auc_pr'], metrics['precision'],
        metrics['recall'], metrics['f1_score'], metrics['specificity']
    ]
    
    bars = axes[1].bar(metric_names, metric_values)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Classification Metrics')
    axes[1].set_ylabel('Score')
    
    # Color bars based on performance
    for bar, value in zip(bars, metric_values):
        if value >= 0.8:
            bar.set_color('green')
        elif value >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig
