import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
import transformers


def compute_clf_metrics(pred: transformers.EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision_scores = metrics.precision_score(labels, preds, average=None)
    recall_scores = metrics.recall_score(labels, preds, average=None)
    f1_scores = metrics.f1_score(labels, preds, average=None)

    metrics_dict = {
        "loss": F.cross_entropy(
            torch.tensor(pred.predictions), torch.tensor(pred.label_ids)
        ),
        "mcc": metrics.matthews_corrcoef(
            labels,
            preds,
        ),
        "f1_macro": metrics.f1_score(labels, preds, average="macro"),
    }

    metrics_dict |= {f"f1_{i}": v for i, v in enumerate(f1_scores)}
    metrics_dict |= {f"prec_{i}": v for i, v in enumerate(precision_scores)}
    metrics_dict |= {f"recall_{i}": v for i, v in enumerate(recall_scores)}

    return metrics_dict
