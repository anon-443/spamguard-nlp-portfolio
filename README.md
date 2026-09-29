# SpamGuard NLP

An end-to-end SMS spam-classification project built around a reproducible, leakage-aware evaluation. It compares word-only TF-IDF against combined word/character TF-IDF with logistic regression, selects a spam decision threshold on validation data under a precision constraint, and serves the selected model through FastAPI. An optional DistilBERT fine-tuning path uses the same duplicate-aware data split for a modern transformer comparison.

## Why this is stronger than a basic classifier

- Uses UCI's **real 5,574-message SMS Spam Collection**, not synthetic examples.
- Groups exact normalized duplicate messages before creating a stratified 60/20/20 train/validation/test split; repeated texts do not cross partitions.
- Compares word n-grams with word + character n-grams, selects by validation average precision, and tunes a high-precision spam threshold using validation only.
- Reports held-out spam precision/recall/F1, macro-F1, average precision, ROC-AUC, Brier score, false-positive rate, duplicate-group bootstrap intervals, and message-length slices.
- Includes an optional DistilBERT comparator and a REST inference API with request validation and Swagger UI.

## Dataset and attribution

The training script downloads the official [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) on first use. UCI reports **5,574** English SMS messages; the verified downloaded file contains **4,827 ham and 747 spam** examples. The text file has one line per message: a `ham`/`spam` label, a tab, and the raw message. The dataset is CC BY 4.0; cite Almeida & Hidalgo (2011), [UCI DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84). The raw dataset is kept under `data/` and excluded from Git and the source archive. See [`SOURCES.md`](SOURCES.md).

## Install and train the lexical baselines

Python 3.10+ is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
spamguard-train --data-file data/raw/SMSSpamCollection --output-dir artifacts/baseline --seed 42 --precision-target 0.95
```

If the data file does not exist, `spamguard-train` fetches and validates it from UCI. The fixed seed makes the grouped split repeatable. Candidate choice is based on validation average precision; the test partition is reserved for final reporting. The spam threshold is the validation threshold with maximum recall among thresholds satisfying the requested validation precision, or a validation-F1 fallback if that constraint cannot be met.

Training writes a joblib model bundle, JSON metrics/configuration, per-class metrics, test predictions without message text, a confusion matrix, PR curve, and calibration plot under the output directory. Raw message content is intentionally not written into metrics or prediction artifacts.

## Reference run on the UCI data

The verified seed-42 run used **3,344 train / 1,115 validation / 1,115 test messages** from **5,159 normalized exact-text groups**. Candidate validation average precision was **0.9776** for word-only TF-IDF and **0.9878** for word + character TF-IDF, so the latter was selected. A threshold of **0.3800** was selected on validation for a target spam precision of 0.95; validation achieved 0.953 spam precision and 0.953 recall.

On the duplicate-group-held-out test set, the selected lexical model achieved **98.74% accuracy**, **0.9727 macro-F1**, **0.9749 average precision**, and **0.9909 ROC-AUC**. At the validation-selected threshold, spam precision was **96.58%**, spam recall **94.00%**, and spam F1 **0.9527**; there were 5 false positives and 9 false negatives among 1,115 test messages. A 1,000-resample bootstrap over normalized exact-text groups gave a 95% interval of **93.1%–99.3%** for spam precision and **89.5%–97.5%** for spam recall. See `artifacts/baseline/metrics.json` and the plots for the full run.

**Important slice limitation:** among the 580 test messages at most 60 characters, only **3 were spam**; the model correctly found 1 and predicted 4 as spam (slice precision 25%, recall 33.3%). That tiny positive support makes the slice estimate noisy, but it flags a genuine risk: this old corpus does not establish reliable performance on short spam. The overall score should not be treated as a modern production guarantee.

## Run the optional transformer comparison

Install PyTorch using the official [PyTorch install selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e '.[transformer]'
spamguard-train-transformer --data-file data/raw/SMSSpamCollection --output-dir artifacts/distilbert --epochs 3 --seed 42
```

The verified three-epoch `distilbert-base-uncased` run used the **same seed-42 duplicate-group split** as the lexical baseline and selected its checkpoint by validation loss (best epoch 2). Both models selected their spam thresholds on validation to target at least 95% precision, then were evaluated once on the same 1,115-message test set.

| Model | Test accuracy | Macro-F1 | Spam precision | Spam recall | Average precision |
|---|---:|---:|---:|---:|---:|
| Word + character TF-IDF logistic regression | 0.9874 | 0.9727 | 0.9658 | 0.9400 | 0.9749 |
| DistilBERT | 0.9910 | 0.9806 | 0.9730 | 0.9600 | 0.9862 |

For DistilBERT, 1,000 normalized-text-group bootstrap resamples gave 95% intervals of **94.1%–99.4%** for spam precision and **92.1%–98.7%** for spam recall. The TF-IDF intervals are **93.1%–99.3%** and **89.5%–97.5%**, respectively; these intervals overlap, so describe the transformer's scores as a nominal improvement, not a statistically proven win. Both models share the same weak short-message slice: of 580 test messages at most 60 characters, only 3 were spam; each found 1 and produced 4 spam alerts. This remains the most important caveat.

The fine-tuned model and tokenizer are saved under `artifacts/distilbert/model`; the training log, metrics, per-class report, plots, and run configuration sit beside it. The FastAPI demo below continues to serve the lightweight TF-IDF model—the transformer is an offline comparison, not the deployed API model. Do not compare with runs using different splits as if they were paired experiments.

## Start the prediction API

After training the lexical model:

```bash
SPAMGUARD_MODEL=artifacts/baseline/model.joblib uvicorn spamguard.api:app --app-dir src --host 0.0.0.0 --port 8019
```

Open `http://localhost:8019/docs` for interactive Swagger UI. `GET /health` reports readiness, and `POST /predict` accepts a JSON object such as `{"text":"Congratulations! Claim your free prize now"}`. The response includes the predicted label, spam probability, validation-selected threshold, and model name. `POST /predict/batch` accepts up to 20 messages. Avoid sending confidential SMS text to any temporary public demo.

Temporary live demo for this task session: [Swagger UI](https://8765-iob7qxpw0ti3uvx48v4ep-a7a4073f.us4.manus.computer/docs) and [health check](https://8765-iob7qxpw0ti3uvx48v4ep-a7a4073f.us4.manus.computer/health). The service is temporary and does not persist after its Sandbox host session is stopped; the local command above is the reproducible deployment path.

## Tests, interpretation, and limitations

```bash
pytest -q
ruff check src tests
```

The test set is held out by normalized exact-message group, but all examples are from an older English SMS collection. Report the dataset and split protocol alongside scores. This is a research/demo filter—not a production message-security control. Do not automatically delete, block, or forward messages based on its output; inspect uncertain cases and establish a current, representative evaluation before any real deployment.

## Resume bullet (use the measured baseline result)

> Built a duplicate-aware SMS spam NLP pipeline comparing word and character TF-IDF logistic models; selected the decision threshold on validation for a 95% spam-precision target, achieved **0.9727 macro-F1** and **96.6% spam precision / 94.0% recall** (group-bootstrap 95% intervals: 93.1%–99.3% precision, 89.5%–97.5% recall) on 1,115 held-out UCI messages, and served predictions through a validated FastAPI API.


### Transformer-focused alternative

> Fine-tuned DistilBERT for SMS spam detection and compared it with word/character TF-IDF using the same normalized-duplicate-grouped split; achieved **0.9806 macro-F1**, **97.3% spam precision**, and **96.0% recall** on 1,115 held-out UCI messages, using a validation-selected threshold and 1,000-group-bootstrap confidence intervals.

When discussing either bullet, call out the old corpus and the low-support short-message slice; do not imply the scores establish current production reliability.
