"""Sparse lexical feature pipelines used as interpretable fast baselines."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="liblinear",
    )


def candidate_pipelines(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "word_tfidf_logistic": Pipeline(
            [("tfidf", word_vectorizer()), ("classifier", classifier(seed))]
        ),
        "word_char_tfidf_logistic": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [("word", word_vectorizer()), ("char", char_vectorizer())],
                        transformer_weights={"word": 1.0, "char": 0.8},
                    ),
                ),
                ("classifier", classifier(seed)),
            ]
        ),
    }
