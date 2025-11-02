import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def log_confusion_matrix(mlflow, y_true, y_pred, name="confusion_matrix.png"):
    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Matriz de confusión")
    mlflow.log_figure(fig, name)
    plt.close(fig)


def log_pr_roc(mlflow, y_true, y_proba):
    if y_proba is None:
        return
    fig1 = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Curva Precision-Recall")
    mlflow.log_figure(fig1, "precision_recall.png")
    plt.close(fig1)

    fig2 = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("Curva ROC")
    mlflow.log_figure(fig2, "roc_curve.png")
    plt.close(fig2)
