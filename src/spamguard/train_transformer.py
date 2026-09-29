"""Optional DistilBERT comparator; install the transformer extra to run it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from spamguard.data import duplicate_aware_split, load_sms, normalize_duplicate_group
from spamguard.metrics import (
    add_length_slices,
    save_per_class_csv,
    save_plots,
    score_predictions,
    select_threshold,
)


def train_transformer(args: argparse.Namespace) -> Path:
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install optional dependencies with: pip install -e '.[transformer]'"
        ) from exc

    data_path = Path(args.data_file)
    messages, labels = load_sms(data_path)
    train_idx, validation_idx, test_idx = duplicate_aware_split(messages, labels, seed=args.seed)
    label_to_id = {"ham": 0, "spam": 1}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label={0: "ham", 1: "spam"}, label2id=label_to_id
    )

    class SMSDataset(Dataset):
        def __init__(self, indices: np.ndarray):
            self.encodings = tokenizer(
                messages[indices].tolist(), truncation=True, max_length=args.max_length
            )
            self.targets = [label_to_id[label] for label in labels[indices]]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.targets[index], dtype=torch.long)
            return item

    def compute_metrics(eval_prediction):
        logits, target_ids = eval_prediction
        predictions = np.argmax(logits, axis=1)
        truth = np.asarray(target_ids)
        return {
            "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
            "accuracy": float(np.mean(truth == predictions)),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=40,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SMSDataset(train_idx),
        eval_dataset=SMSDataset(validation_idx),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    validation_logits = trainer.predict(SMSDataset(validation_idx)).predictions
    validation_probability = torch.softmax(torch.as_tensor(validation_logits), dim=1)[:, 1].numpy()
    threshold, threshold_policy = select_threshold(
        labels[validation_idx], validation_probability, args.precision_target
    )
    test_logits = trainer.predict(SMSDataset(test_idx)).predictions
    test_probability = torch.softmax(torch.as_tensor(test_logits), dim=1)[:, 1].numpy()
    test_groups = np.asarray(
        [normalize_duplicate_group(text) for text in messages[test_idx]], dtype=str
    )
    metrics, prediction = score_predictions(
        labels[test_idx], test_probability, threshold, seed=args.seed, groups=test_groups
    )
    add_length_slices(metrics, messages[test_idx], labels[test_idx], prediction)
    metrics.update(
        {
            "model": args.base_model,
            "dataset": "UCI SMS Spam Collection, DOI 10.24432/C5CC84",
            "seed": args.seed,
            "decision_threshold": threshold,
            "threshold_policy": threshold_policy,
            "validation_precision_target": args.precision_target,
            "train_examples": len(train_idx),
            "validation_examples": len(validation_idx),
            "test_examples": len(test_idx),
            "split_protocol": "stratified grouped folds over normalized exact-message text",
        }
    )
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_per_class_csv(metrics, output_dir / "per_class_metrics.csv")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    save_plots(labels[test_idx], test_probability, prediction, output_dir)
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if not key.endswith("_ci")}, indent=2
        )
    )
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="data/raw/SMSSpamCollection")
    parser.add_argument("--output-dir", default="artifacts/distilbert")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision-target", type=float, default=0.95)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.precision_target < 1:
        raise SystemExit("epochs/batch-size must be positive and precision-target between 0 and 1")
    train_transformer(args)


if __name__ == "__main__":
    main()
