import numpy as np
import pytest

from spamguard.api import TextRequest, predict_one
from spamguard.data import duplicate_aware_split, load_sms, normalize_duplicate_group
from spamguard.metrics import score_predictions, select_threshold
from spamguard.models import candidate_pipelines


def test_load_sms_parses_tabs_and_keeps_message_punctuation(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    path.write_text("ham\tHello there!\nspam\tWIN £500 now!\n", encoding="utf-8")
    messages, labels = load_sms(path)
    assert messages.tolist() == ["Hello there!", "WIN £500 now!"]
    assert labels.tolist() == ["ham", "spam"]


def test_load_sms_rejects_malformed_rows(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("unknown\tmessage\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid label"):
        load_sms(path)


def test_duplicate_aware_split_keeps_repeated_text_groups_together():
    messages = [f"ham sample number {i}" for i in range(20)] + [
        f"spam offer code {i}" for i in range(20)
    ]
    labels = ["ham"] * 20 + ["spam"] * 20
    messages += [messages[0].upper(), "  " + messages[25] + "  "]
    labels += ["ham", "spam"]
    messages = np.asarray(messages)
    labels = np.asarray(labels)
    train, validation, test = duplicate_aware_split(messages, labels, seed=17)
    groups = [
        {normalize_duplicate_group(messages[index]) for index in indices}
        for indices in (train, validation, test)
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    for indices in (train, validation, test):
        assert set(labels[indices]) == {"ham", "spam"}


def test_validation_threshold_and_metrics():
    truth = np.asarray(["ham"] * 8 + ["spam"] * 4)
    probability = np.asarray([0.03, 0.08, 0.1, 0.12, 0.15, 0.2, 0.22, 0.3, 0.7, 0.8, 0.88, 0.96])
    threshold, policy = select_threshold(truth, probability, precision_target=0.9)
    assert 0.0 <= threshold <= 1.0
    assert policy == "max_validation_recall_at_precision_target"
    groups = np.asarray([f"text_{index // 2}" for index in range(len(truth))])
    metrics, prediction = score_predictions(truth, probability, threshold, groups=groups)
    assert len(prediction) == len(truth)
    assert 0 <= metrics["spam_precision"] <= 1
    assert 0 <= metrics["spam_recall"] <= 1
    assert metrics["confusion_matrix"][0][0] + metrics["confusion_matrix"][0][1] == 8
    assert "spam_recall_95pct_bootstrap_ci" in metrics
    assert metrics["bootstrap_unit"] == "normalized_exact_text_group"


def test_candidate_pipelines_have_independent_vectorizers():
    candidates = candidate_pipelines()
    word = candidates["word_tfidf_logistic"].named_steps["tfidf"]
    word_char = (
        candidates["word_char_tfidf_logistic"].named_steps["features"].transformer_list[0][1]
    )
    assert word is not word_char


def test_api_validation_and_prediction_payload():
    with pytest.raises(ValueError):
        TextRequest(text="   ")

    class DummyPipeline:
        classes_ = np.asarray(["ham", "spam"])

        def predict_proba(self, texts):
            return np.asarray([[0.1, 0.9] for _ in texts])

    result = predict_one(
        "claim a free prize",
        {
            "pipeline": DummyPipeline(),
            "class_order": ["ham", "spam"],
            "decision_threshold": 0.8,
            "model_name": "test-double",
        },
    )
    assert result["label"] == "spam"
    assert result["is_spam"] is True
    assert result["spam_probability"] == pytest.approx(0.9)
