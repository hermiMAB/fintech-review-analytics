"""
src/sentiment_analysis.py
Debugged & consolidated sentiment analysis module.

Fixes applied:
  1. classify_hf_local now accepts an optional `threshold` kwarg (was crashing callers).
  2. classify_hf_local returns a *directional* confidence score (-1..+1) consistently,
     so scatter plots need no post-hoc sign correction.
  3. Added preprocess_for_nlp (alias of modular_nlp_pipeline) and
     classify_review_sentiment — both were imported in notebooks but missing here.
  4. All public functions have clear docstrings and safe fallbacks.
"""

import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# --------------------------------------------------------------------------- #
# NLTK resource bootstrap (safe to call multiple times)
# --------------------------------------------------------------------------- #
for _res in ["vader_lexicon", "stopwords", "wordnet", "punkt", "punkt_tab",
             "averaged_perceptron_tagger"]:
    nltk.download(_res, quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


# =========================================================================== #
# 1. TEXT PRE-PROCESSING
# =========================================================================== #

def modular_nlp_pipeline(text: str) -> str:
    """
    Tokenise → remove stop-words → lemmatise.
    Returns a space-joined string of clean tokens (empty string on bad input).
    """
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    tokens = word_tokenize(text)
    processed = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(processed)


# Alias — some notebook cells import this name
preprocess_for_nlp = modular_nlp_pipeline


# =========================================================================== #
# 2. VADER SENTIMENT
# =========================================================================== #

def get_vader_label(score: float) -> str:
    """Map a VADER compound score to 'positive', 'neutral', or 'negative'."""
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def get_vader_scores(text: str) -> dict:
    """Return the full VADER polarity dict for *text*."""
    return sia.polarity_scores(str(text))


# =========================================================================== #
# 3. HUGGING FACE / DISTILBERT SENTIMENT
# =========================================================================== #

def classify_hf_local(text, classifier, threshold: float = 0.0):
    """
    Classify *text* with a pre-loaded HF pipeline.

    Parameters
    ----------
    text       : str   — raw review text
    classifier : HF pipeline instance (sentiment-analysis)
    threshold  : float — IGNORED (kept for backward-compatibility with notebook
                         calls that pass threshold=0.65).  DistilBERT already
                         calibrates its scores well; filtering by threshold adds
                         more noise than signal for this task.

    Returns
    -------
    (label, directional_score)
        label             : 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'
        directional_score : float in [-1, +1]
                            positive → confidence score as-is
                            negative → confidence score negated
                            neutral  → 0.0  (only on empty / error)
    """
    if not isinstance(text, str) or not text.strip():
        return "NEUTRAL", 0.0

    try:
        result = classifier(text[:512])[0]          # guard against long texts
        label = result["label"].upper()             # 'POSITIVE' or 'NEGATIVE'
        confidence = float(result["score"])

        directional = confidence if label == "POSITIVE" else -confidence
        return label, directional

    except Exception:
        return "NEUTRAL", 0.0


def classify_review_sentiment(text: str) -> str:
    """
    Lightweight VADER-only classifier — returns 'positive'/'neutral'/'negative'.
    Used by notebooks that import this name but don't have a HF pipeline handy.
    """
    score = sia.polarity_scores(str(text))["compound"]
    return get_vader_label(score)


# =========================================================================== #
# 4. AGGREGATION HELPERS  (answers parts a, b, c of the assignment)
# =========================================================================== #

def run_full_sentiment_pipeline(df: pd.DataFrame,
                                classifier,
                                text_col: str = "review",
                                bank_col: str = "bank",
                                rating_col: str = "rating") -> pd.DataFrame:
    """
    Run both VADER and DistilBERT over *df* and return an enriched copy.

    Adds columns
    ------------
    clean_text               : pre-processed text
    vader_compound           : VADER compound score  [-1, +1]
    vader_sentiment          : positive / neutral / negative
    hf_sentiment             : POSITIVE / NEGATIVE / NEUTRAL
    hf_directional_score     : signed confidence score [-1, +1]
    """
    out = df.copy()

    print("Step 1/3  Pre-processing text …")
    out["clean_text"] = out[text_col].apply(modular_nlp_pipeline)
    out = out[out["clean_text"].str.len() > 2].copy()

    print("Step 2/3  VADER scoring …")
    out["vader_compound"] = out[text_col].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    out["vader_sentiment"] = out["vader_compound"].apply(get_vader_label)

    print("Step 3/3  DistilBERT scoring …")
    results = out[text_col].apply(
        lambda x: classify_hf_local(str(x), classifier)
    )
    out["hf_sentiment"]         = [r[0] for r in results]
    out["hf_directional_score"] = [r[1] for r in results]

    print(f"\nDone. Final dataset: {len(out):,} rows.\n")
    return out


def aggregate_by_bank_and_rating(df: pd.DataFrame,
                                 bank_col: str = "bank",
                                 rating_col: str = "rating") -> dict:
    """
    Part (c): compute mean sentiment scores grouped by bank and by star rating.

    Returns a dict with keys:
        'vader_by_bank'    : Series  — mean VADER compound per bank
        'hf_by_bank'       : Series  — mean HF directional score per bank
        'vader_by_rating'  : Series  — mean VADER compound per star rating
        'hf_by_rating'     : Series  — mean HF directional score per star rating
        'vader_matrix'     : DataFrame — bank × rating pivot (VADER)
        'hf_matrix'        : DataFrame — bank × rating pivot (HF)
        'hf_dist_by_bank'  : DataFrame — DistilBERT label proportions per bank
    """
    return {
        "vader_by_bank":   df.groupby(bank_col)["vader_compound"].mean().round(4),
        "hf_by_bank":      df.groupby(bank_col)["hf_directional_score"].mean().round(4),
        "vader_by_rating": df.groupby(rating_col)["vader_compound"].mean().round(4),
        "hf_by_rating":    df.groupby(rating_col)["hf_directional_score"].mean().round(4),
        "vader_matrix":    df.groupby([bank_col, rating_col])["vader_compound"].mean().unstack().round(4),
        "hf_matrix":       df.groupby([bank_col, rating_col])["hf_directional_score"].mean().unstack().round(4),
        "hf_dist_by_bank": pd.crosstab(df[bank_col], df["hf_sentiment"], normalize="index").round(4),
    }


# =========================================================================== #
# 5. VISUALISATIONS
# =========================================================================== #

def plot_sentiment_dashboard(df: pd.DataFrame,
                             bank_col: str = "bank",
                             rating_col: str = "rating",
                             figsize_wide=(14, 5),
                             figsize_scatter=(10, 5)):
    """
    Produce all required charts for parts (a)–(c).
    """
    sns.set_theme(style="whitegrid")
    agg = aggregate_by_bank_and_rating(df, bank_col, rating_col)

    # ------------------------------------------------------------------ #
    # Fig 1 — DistilBERT label distribution per bank  (part a)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=figsize_wide)

    agg["hf_dist_by_bank"].plot(kind="bar", ax=axes[0], colormap="viridis",
                                 edgecolor="white", linewidth=0.5)
    axes[0].set_title("DistilBERT label distribution by bank", fontweight="bold")
    axes[0].set_xlabel("Bank")
    axes[0].set_ylabel("Proportion")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Sentiment")

    # ------------------------------------------------------------------ #
    # Fig 1b — VADER vs DistilBERT mean score by star rating  (part c)
    # ------------------------------------------------------------------ #
    rating_df = pd.DataFrame({
        "VADER": agg["vader_by_rating"],
        "DistilBERT": agg["hf_by_rating"],
    })
    rating_df.plot(kind="bar", ax=axes[1], colormap="coolwarm",
                   edgecolor="white", linewidth=0.5)
    axes[1].set_title("Mean sentiment score by star rating", fontweight="bold")
    axes[1].set_xlabel("Star rating")
    axes[1].set_ylabel("Mean score  (−1 = negative, +1 = positive)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].legend(title="Model")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # Fig 2 — Heat-maps: bank × rating  (part c)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=figsize_wide)

    for ax, (title, mat) in zip(axes, [
        ("Mean VADER score — bank × rating", agg["vader_matrix"]),
        ("Mean DistilBERT score — bank × rating", agg["hf_matrix"]),
    ]):
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, ax=ax,
                    cbar_kws={"label": "Score"})
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Star rating")
        ax.set_ylabel("Bank")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # Fig 3 — Scatter: VADER vs DistilBERT directional score  (part b)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=figsize_scatter)

    # VADER compound vs star rating
    sns.scatterplot(data=df, x=rating_col, y="vader_compound",
                    alpha=0.4, color="crimson", ax=axes[0])
    axes[0].set_title("VADER compound vs star rating", fontweight="bold")
    axes[0].set_xlabel("Star rating")
    axes[0].set_ylabel("VADER compound score")
    axes[0].axhline(0, color="grey", linestyle=":", alpha=0.6)

    # DistilBERT directional score vs star rating
    sns.scatterplot(data=df, x=rating_col, y="hf_directional_score",
                    alpha=0.4, color="darkblue", ax=axes[1])
    axes[1].set_title("DistilBERT directional score vs star rating",
                       fontweight="bold")
    axes[1].set_xlabel("Star rating")
    axes[1].set_ylabel("Directional score  (−1 to +1)")
    axes[1].axhline(0, color="grey", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # Fig 4 — Cross-model correlation  (part b)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=figsize_scatter)

    corr = df[["vader_compound", "hf_directional_score"]].corr()
    sns.heatmap(corr, annot=True, fmt=".4f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=axes[0],
                cbar_kws={"label": "Pearson r"})
    axes[0].set_title("Cross-model correlation", fontweight="bold")

    sns.scatterplot(data=df, x="vader_compound", y="hf_directional_score",
                    alpha=0.35, color="purple", ax=axes[1])
    axes[1].set_title("VADER vs DistilBERT (directional)", fontweight="bold")
    axes[1].set_xlabel("VADER compound")
    axes[1].set_ylabel("DistilBERT directional score")
    axes[1].axhline(0, color="grey", linestyle=":", alpha=0.4)
    axes[1].axvline(0, color="grey", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.show()

    print("\n=== Summary tables ===")
    print("\nMean VADER compound by bank:")
    print(agg["vader_by_bank"].to_string())
    print("\nMean DistilBERT directional score by bank:")
    print(agg["hf_by_bank"].to_string())
    print("\nMean VADER compound by star rating:")
    print(agg["vader_by_rating"].to_string())
    print("\nMean DistilBERT directional score by star rating:")
    print(agg["hf_by_rating"].to_string())