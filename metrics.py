from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)

def calculate_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": p,
        "recall_weighted": r,
        "f1_weighted": f1,
        "f1_macro": macro_f1,
        "report": classification_report(
            y_true, y_pred, zero_division=0
        )
    }
