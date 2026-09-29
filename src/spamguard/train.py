"""Compare sparse lexical models on UCI SMS data with duplicate-aware evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

from spamguard.data import download_uci, duplicate_aware_split, load_sms, normalize_duplicate_group
from spamguard.metrics import (
    add_length_slices,
    save_per_class_csv,
    save_plots,
    score_predictions,
    select_threshold,
)
from spamguard.models import candidate_pipelines


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_file = download_uci(args.data_file)
    messages, labels = load_sms(data_file)
    train_idx, validation_idx, test_idx = duplicate_aware_split(messages, labels, seed=args.seed)
    candidates = candidate_pipelines(args.seed)
    validation_results = {}
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(messages[train_idx].tolist(), labels[train_idx].tolist())
        spam_column = list(pipeline.classes_).index("spam")
        validation_probability = pipeline.predict_proba(messages[validation_idx].tolist())[
            :, spam_column
        ]
        average_precision = float(
            average_precision_score(labels[validation_idx] == "spam", validation_probability)
        )
        validation_results[name] = {
            "validation_average_precision": average_precision,
            "validation_spam_precision_at_0_5": float(
                precision_score(
                    labels[validation_idx],
                    np.where(validation_probability >= 0.5, "spam", "ham"),
                    pos_label="spam",
                    zero_division=0,
                )
            ),
            "validation_spam_recall_at_0_5": float(
                recall_score(
                    labels[validation_idx],
                    np.where(validation_probability >= 0.5, "spam", "ham"),
                    pos_label="spam",
                    zero_division=0,
                )
            ),
        }
        fitted[name] = (pipeline, validation_probability)
        print(f"candidate={name} validation_average_precision={average_precision:.4f}", flush=True)

    selected_name = max(
        validation_results,
        key=lambda name: validation_results[name]["validation_average_precision"],
    )
    model, validation_probability = fitted[selected_name]
    threshold, threshold_policy = select_threshold(
        labels[validation_idx], validation_probability, args.precision_target
    )
    spam_column = list(model.classes_).index("spam")
    test_probability = model.predict_proba(messages[test_idx].tolist())[:, spam_column]
    test_groups = np.asarray(
        [normalize_duplicate_group(text) for text in messages[test_idx]], dtype=str
    )
    test_metrics, test_prediction = score_predictions(
        labels[test_idx], test_probability, threshold, seed=args.seed, groups=test_groups
    )
    add_length_slices(test_metrics, messages[test_idx], labels[test_idx], test_prediction)
    test_metrics.update(
        {
            "selected_model": selected_name,
            "selection_metric": "validation average precision",
            "validation_candidate_results": validation_results,
            "decision_threshold": threshold,
            "threshold_policy": threshold_policy,
            "validation_precision_target": args.precision_target,
            "test_examples": len(test_idx),
            "train_examples": len(train_idx),
            "validation_examples": len(validation_idx),
            "total_examples": len(messages),
            "observed_label_counts": dict(Counter(labels.tolist())),
            "unique_normalized_text_groups": len(
                {normalize_duplicate_group(text) for text in messages}
            ),
            "split_protocol": "stratified grouped folds over normalized exact-message text; no duplicate group crosses partitions",
            "seed": args.seed,
            "dataset": "UCI SMS Spam Collection, DOI 10.24432/C5CC84",
        }
    )
    validation_decision = np.where(validation_probability >= threshold, "spam", "ham")
    test_metrics["validation_spam_precision_at_selected_threshold"] = float(
        precision_score(
            labels[validation_idx], validation_decision, pos_label="spam", zero_division=0
        )
    )
    test_metrics["validation_spam_recall_at_selected_threshold"] = float(
        recall_score(labels[validation_idx], validation_decision, pos_label="spam", zero_division=0)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "pipeline": model,
            "model_name": selected_name,
            "decision_threshold": threshold,
            "positive_label": "spam",
            "class_order": list(model.classes_),
            "seed": args.seed,
            "dataset_doi": "10.24432/C5CC84",
        },
        model_path,
        compress=3,
    )
    (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    save_per_class_csv(test_metrics, output_dir / "per_class_metrics.csv")
    run_config = {
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "precision_target": args.precision_target,
        "group_split_sizes": {
            "train": len(train_idx),
            "validation": len(validation_idx),
            "test": len(test_idx),
        },
        "duplicate_group_counts": {
            "train": len({normalize_duplicate_group(messages[i]) for i in train_idx}),
            "validation": len({normalize_duplicate_group(messages[i]) for i in validation_idx}),
            "test": len({normalize_duplicate_group(messages[i]) for i in test_idx}),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_row_index", "true_label", "predicted_label", "spam_probability"])
        for row, truth, prediction, probability in zip(
            test_idx, labels[test_idx], test_prediction, test_probability, strict=True
        ):
            writer.writerow([int(row), truth, prediction, float(probability)])
    save_plots(labels[test_idx], test_probability, test_prediction, output_dir)
    print(
        json.dumps(
            {
                key: value
                for key, value in test_metrics.items()
                if not key.startswith("short_")
                and not key.startswith("medium_")
                and not key.startswith("long_")
            },
            indent=2,
        )
    )
    print(f"saved_model={model_path}")
    return model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="data/raw/SMSSpamCollection")
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision-target", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.precision_target < 1:
        raise SystemExit("precision-target must be between 0 and 1")
    train(args)


if __name__ == "__main__":
    main()
