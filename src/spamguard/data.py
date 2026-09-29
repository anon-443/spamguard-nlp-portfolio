"""UCI corpus acquisition, parsing, and duplicate-aware data splitting."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_uci(data_file: str | Path) -> Path:
    destination = Path(data_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    archive_path = destination.parent / "uci_sms_spam_collection.zip"
    with urlopen(UCI_URL, timeout=60) as response:
        archive_path.write_bytes(response.read())
    with ZipFile(archive_path) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith("SMSSpamCollection")), None
        )
        if member is None:
            raise ValueError("UCI archive did not contain SMSSpamCollection")
        destination.write_bytes(archive.read(member))
    return destination


def load_sms(data_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(data_file)
    if not path.exists():
        download_uci(path)
    messages, labels = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label, message = line.lstrip("\ufeff").split("\t", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed UCI row {line_number}: expected tab-separated label and text"
            ) from exc
        if label not in {"ham", "spam"} or not message.strip():
            raise ValueError(f"Malformed UCI row {line_number}: invalid label or empty message")
        messages.append(message.strip())
        labels.append(label)
    if not messages:
        raise ValueError(f"No messages were parsed from {path}")
    return np.asarray(messages, dtype=str), np.asarray(labels, dtype=str)


def normalize_duplicate_group(text: str) -> str:
    """Case-fold and collapse whitespace to keep repeated SMS texts in one split."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def duplicate_aware_split(
    messages: np.ndarray, labels: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test indices with exact normalized texts group-isolated.

    Uses one fold of stratified grouped 5-fold CV for test (about 20%), then one
    fold of grouped 4-fold CV on the remainder for validation (about 20% overall).
    """
    messages = np.asarray(messages, dtype=str)
    labels = np.asarray(labels, dtype=str)
    if messages.ndim != 1 or labels.ndim != 1 or len(messages) != len(labels):
        raise ValueError("messages and labels must be one-dimensional arrays of equal length")
    if len(np.unique(labels)) != 2:
        raise ValueError("the classifier expects exactly two classes")
    groups = np.asarray([normalize_duplicate_group(message) for message in messages], dtype=str)
    if np.any(groups == ""):
        raise ValueError("messages must be non-empty")
    indices = np.arange(len(labels))
    train_val, test = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
            indices, labels, groups
        )
    )
    train_relative, validation_relative = next(
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1).split(
            train_val, labels[train_val], groups[train_val]
        )
    )
    train = train_val[train_relative]
    validation = train_val[validation_relative]
    partition_groups = [set(groups[part]) for part in (train, validation, test)]
    if any(partition_groups[i] & partition_groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("duplicate-message leakage detected between partitions")
    expected = set(np.unique(labels))
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        missing = expected - set(np.unique(labels[part]))
        if missing:
            raise ValueError(
                f"{name} split is missing class(es) {sorted(missing)}; try another seed"
            )
    return train, validation, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify UCI SMS Spam Collection")
    parser.add_argument("--output", default="data/raw/SMSSpamCollection")
    args = parser.parse_args()
    path = download_uci(args.output)
    messages, labels = load_sms(path)
    values, counts = np.unique(labels, return_counts=True)
    print(f"Downloaded {len(messages)} messages to {path}")
    print("Label counts:", dict(zip(values.tolist(), counts.tolist(), strict=True)))


if __name__ == "__main__":
    main()
