"""Threshold policy, held-out classification metrics, and diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = "spam"


def select_threshold(
    y_true: np.ndarray, spam_probability: np.ndarray, precision_target: float
) -> tuple[float, str]:
    if not 0 < precision_target < 1:
        raise ValueError("precision_target must be between 0 and 1")
    precision, recall, thresholds = precision_recall_curve(
        y_true, spam_probability, pos_label=POSITIVE_LABEL
    )
    eligible = np.flatnonzero(precision[:-1] >= precision_target)
    if eligible.size:
        index = int(eligible[np.argmax(recall[eligible])])
        return float(thresholds[index]), "max_validation_recall_at_precision_target"
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(f1))
    return float(thresholds[index]), "validation_f1_fallback_target_unmet"


def bootstrap_spam_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    seed: int = 42,
    iterations: int = 1000,
) -> dict:
    rng = np.random.default_rng(seed)
    grouped_bootstrap = groups is not None
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    if len(groups) != len(y_true):
        raise ValueError("bootstrap groups must have the same length as labels")
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    precision_values, recall_values = [], []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        truth, predicted = y_true[indices], y_pred[indices]
        tp = np.count_nonzero((truth == POSITIVE_LABEL) & (predicted == POSITIVE_LABEL))
        fp = np.count_nonzero((truth != POSITIVE_LABEL) & (predicted == POSITIVE_LABEL))
        fn = np.count_nonzero((truth == POSITIVE_LABEL) & (predicted != POSITIVE_LABEL))
        precision_values.append(tp / (tp + fp) if tp + fp else 0.0)
        recall_values.append(tp / (tp + fn) if tp + fn else 0.0)
    return {
        "spam_precision_95pct_bootstrap_ci": [
            float(v) for v in np.percentile(precision_values, [2.5, 97.5])
        ],
        "spam_recall_95pct_bootstrap_ci": [
            float(v) for v in np.percentile(recall_values, [2.5, 97.5])
        ],
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "bootstrap_unit": "normalized_exact_text_group" if grouped_bootstrap else "example",
    }


def score_predictions(
    y_true: np.ndarray,
    spam_probability: np.ndarray,
    threshold: float,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    y_true = np.asarray(y_true, dtype=str)
    spam_probability = np.asarray(spam_probability, dtype=float)
    if len(y_true) != len(spam_probability) or not np.isfinite(spam_probability).all():
        raise ValueError("truth and finite probability arrays must have equal lengths")
    y_pred = np.where(spam_probability >= threshold, "spam", "ham")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).ravel()
    y_binary = (y_true == POSITIVE_LABEL).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=["ham", "spam"], average="macro", zero_division=0)
        ),
        "spam_precision": float(precision_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "spam_recall": float(recall_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "spam_f1": float(f1_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "average_precision": float(average_precision_score(y_binary, spam_probability)),
        "roc_auc": float(roc_auc_score(y_binary, spam_probability)),
        "brier_score": float(brier_score_loss(y_binary, spam_probability)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "per_class": classification_report(
            y_true, y_pred, labels=["ham", "spam"], output_dict=True, zero_division=0
        ),
    }
    metrics.update(bootstrap_spam_intervals(y_true, y_pred, groups=groups, seed=seed))
    return metrics, y_pred


def save_per_class_csv(metrics: dict, output_path: Path) -> None:
    import csv

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["class", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()
        for label in ("ham", "spam"):
            row = metrics["per_class"][label]
            writer.writerow({"class": label, **row})


def add_length_slices(
    metrics: dict, messages: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    lengths = np.asarray([len(message) for message in messages])
    for name, mask in (
        ("short_le_60_chars", lengths <= 60),
        ("medium_61_to_120_chars", (lengths > 60) & (lengths <= 120)),
        ("long_gt_120_chars", lengths > 120),
    ):
        if mask.any():
            truth, prediction = y_true[mask], y_pred[mask]
            metrics[name] = {
                "count": int(mask.sum()),
                "spam_precision": float(
                    precision_score(truth, prediction, pos_label="spam", zero_division=0)
                ),
                "spam_recall": float(
                    recall_score(truth, prediction, pos_label="spam", zero_division=0)
                ),
            }


def save_plots(
    y_true: np.ndarray, spam_probability: np.ndarray, y_pred: np.ndarray, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Duplicate-group-held-out test",
    )
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, spam_probability, pos_label="spam")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color="#2563eb", linewidth=2)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Spam recall",
        ylabel="Spam precision",
        title="Held-out precision–recall curve",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(
        (y_true == "spam").astype(int), spam_probability, n_bins=8, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", color="#7c3aed", label="logistic model")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted spam probability",
        ylabel="Observed spam fraction",
        title="Held-out reliability diagram",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=180)
    plt.close(fig)
