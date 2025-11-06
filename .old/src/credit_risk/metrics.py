from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    if y_proba is not None and len(set(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            pass
    return out
