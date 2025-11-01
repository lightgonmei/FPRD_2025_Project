"""
src/evaluation_metrics.py

Provides functions to evaluate a trained model and plot results:
- prints accuracy, precision, recall, f1
- plots confusion matrix and ROC curve
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Matplotlib tweaks: do not set colors explicitly (leave defaults)
plt.rcParams.update({'figure.max_open_warning': 0})


def evaluate_and_plot(model_pipeline, X_test, y_test, show_plots: bool = True, save_plots: bool = False, out_dir: str = 'reports'):
    """
    Evaluate the model on X_test / y_test, print metrics, and plot:
      - Confusion matrix
      - ROC curve
    Returns a dict with computed metrics.
    """
    if hasattr(model_pipeline, 'predict_proba'):
        y_score = model_pipeline.predict_proba(X_test)[:, 1]
    else:
        # If no predict_proba, use decision_function if available, else fallback to predictions
        y_score = model_pipeline.decision_function(X_test) if hasattr(model_pipeline, 'decision_function') else model_pipeline.predict(X_test)

    y_pred = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n📈 Evaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)'], zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    if show_plots or save_plots:
        if save_plots:
            os.makedirs(out_dir, exist_ok=True)

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, interpolation='nearest')
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xlabel('Predicted label')
        ax_cm.set_ylabel('True label')
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['Real', 'Fake'])
        ax_cm.set_yticklabels(['Real', 'Fake'])

        # Annotate cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))

        if save_plots:
            fig_cm.savefig(os.path.join(out_dir, 'confusion_matrix.png'), bbox_inches='tight')

        if show_plots:
            plt.show()
        else:
            plt.close(fig_cm)

    # ROC Curve & AUC
    try:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        if show_plots or save_plots:
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=1)
            ax_roc.set_xlim([-0.02, 1.02])
            ax_roc.set_ylim([-0.02, 1.02])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC)')
            ax_roc.legend(loc="lower right")

            if save_plots:
                fig_roc.savefig(os.path.join(out_dir, 'roc_curve.png'), bbox_inches='tight')

            if show_plots:
                plt.show()
            else:
                plt.close(fig_roc)
    except Exception as e:
        print("⚠️ Could not compute ROC curve:", e)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return metrics
