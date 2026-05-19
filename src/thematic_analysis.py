"""
src/thematic_analysis.py
========================
Thematic Analysis module for bank app review data.

Covers all assignment requirements:
  a) Theme definitions with business-relevant categories
  b) TF-IDF keyword & n-gram extraction per bank
  c) Supervised keyword → theme mapping (3–5 themes per bank)
  d) Optional LDA topic modelling for unsupervised theme discovery

Public API
----------
THEME_MAP               : dict  — master keyword → theme registry
assign_themes()         : str   → comma-separated theme labels for one review
extract_keywords_tfidf(): DataFrame of top TF-IDF keywords per bank
extract_top_ngrams()    : DataFrame of top n-grams for a corpus
run_lda()               : fitted LDA model + top-words DataFrame
plot_themes_per_bank()  : stacked-bar chart of theme distribution
plot_theme_sentiment()  : horizontal bar of mean VADER score per theme
plot_lda_topics()       : bar chart grid of LDA topic word distributions
plot_keyword_heatmap()  : heatmap of TF-IDF keyword weights across banks
plot_wordcloud_per_bank(): one word-cloud per bank

Usage
-----
from src.thematic_analysis import (
    assign_themes, extract_keywords_tfidf, extract_top_ngrams,
    run_lda, plot_themes_per_bank, plot_theme_sentiment,
    plot_lda_topics, plot_keyword_heatmap, plot_wordcloud_per_bank,
)
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


# ============================================================================
# a)  THEME DEFINITIONS
# ============================================================================
#
# Grouping logic (document this in your markdown cell):
#   1. Account Access      — authentication, login, OTP, password issues
#   2. Transaction Perf.   — transfers, payments, speed, limits, fees
#   3. App Stability       — crashes, bugs, slow loading, errors
#   4. UI & Design         — interface quality, navigation, ease-of-use
#   5. Customer Support    — help desk, response time, complaint resolution
#   6. Feature Requests    — missing features, service gaps users ask for
#
# A single review can belong to multiple themes (multi-label).
# The keyword lists were built by:
#   (i)  inspecting the top-50 TF-IDF tokens across all three banks, and
#   (ii) augmenting with domain knowledge of Ethiopian fintech pain-points.

THEME_MAP: dict[str, list[str]] = {
    "Account Access": [
        "login", "log", "signin", "sign", "otp", "password", "pasword",
        "verify", "verification", "account", "blocked", "lock", "unlock",
        "suspend", "suspended", "access", "credential", "forgot", "reset",
        "authentication", "fingerprint", "biometric", "face",
    ],
    "Transaction Performance": [
        "transfer", "send", "receive", "payment", "pay", "transaction",
        "delay", "slow", "fast", "speed", "fail", "failed", "pending",
        "limit", "fee", "charge", "debit", "credit", "balance", "fund",
        "money", "cash", "bank", "wire", "remit", "deposit", "withdraw",
        "timeout", "incomplete", "stuck",
    ],
    "App Stability": [
        "crash", "crashing", "freeze", "frozen", "bug", "error",
        "glitch", "problem", "issue", "fix", "update", "version",
        "load", "loading", "blank", "black", "screen", "restart",
        "uninstall", "install", "force", "close", "open", "launch",
        "hang", "response", "not working", "stop",
    ],
    "UI & Design": [
        "ui", "interface", "design", "layout", "navigation", "menu",
        "button", "easy", "simple", "clean", "beautiful", "ugly",
        "confusing", "intuitive", "user", "friendly", "experience",
        "dashboard", "look", "feel", "color", "font", "icon",
        "dark mode", "theme", "modern", "outdated",
    ],
    "Customer Support": [
        "support", "service", "customer", "help", "response", "agent",
        "call", "hotline", "contact", "complaint", "feedback", "resolve",
        "solved", "answer", "reply", "chat", "ticket", "escalate",
        "wait", "waiting", "ignored", "rude", "polite", "helpful",
    ],
    "Feature Requests": [
        "feature", "request", "add", "wish", "want", "need", "missing",
        "please", "hope", "suggest", "suggestion", "improve", "improvement",
        "new", "option", "functionality", "ability", "allow", "support",
        "integrate", "international", "airtime", "utility", "merchant",
    ],
}

# Colour palette — one colour per theme (used consistently across all plots)
THEME_COLORS: dict[str, str] = {
    "Account Access":          "#4C72B0",
    "Transaction Performance": "#DD8452",
    "App Stability":           "#55A868",
    "UI & Design":             "#C44E52",
    "Customer Support":        "#8172B3",
    "Feature Requests":        "#937860",
    "Other":                   "#9E9E9E",
}


# ============================================================================
# b & c)  KEYWORD / N-GRAM EXTRACTION + THEME ASSIGNMENT
# ============================================================================

def assign_themes(text: str, theme_map: dict | None = None) -> str:
    """
    Multi-label theme assignment for a single review string.

    Returns
    -------
    Comma-separated theme names, e.g. "Account Access, App Stability".
    Returns "Other" when no keyword matches.
    """
    if theme_map is None:
        theme_map = THEME_MAP

    if not isinstance(text, str) or not text.strip():
        return "Other"

    text_lower = text.lower()
    matched = [
        theme for theme, keywords in theme_map.items()
        if any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords)
    ]
    return ", ".join(matched) if matched else "Other"


def explode_themes(df: pd.DataFrame,
                   theme_col: str = "themes",
                   bank_col: str = "bank") -> pd.DataFrame:
    """
    Explode multi-label theme column so each row has exactly one theme.
    Useful for groupby / crosstab operations.
    """
    out = df.copy()
    out[theme_col] = out[theme_col].str.split(", ")
    return out.explode(theme_col).reset_index(drop=True)


def extract_keywords_tfidf(df: pd.DataFrame,
                            text_col: str = "clean_text",
                            bank_col: str = "bank",
                            top_n: int = 15,
                            ngram_range: tuple = (1, 2)) -> pd.DataFrame:
    """
    Compute TF-IDF keywords for each bank independently.

    Returns
    -------
    Wide DataFrame — index = bank names, columns = ['Top Keywords', 'Top Phrases']
    where 'Top Keywords' are unigrams and 'Top Phrases' are bigrams.
    """
    rows = []
    for bank, grp in df.groupby(bank_col):
        corpus = grp[text_col].dropna().tolist()
        if not corpus:
            continue

        # Unigrams
        uni_vec = TfidfVectorizer(max_features=100, ngram_range=(1, 1),
                                   stop_words="english")
        uni_mat = uni_vec.fit_transform(corpus)
        uni_scores = uni_mat.mean(axis=0).A1
        uni_top = [
            uni_vec.get_feature_names_out()[i]
            for i in uni_scores.argsort()[::-1][:top_n]
        ]

        # Bigrams
        bi_vec = TfidfVectorizer(max_features=100, ngram_range=(2, 2),
                                  stop_words="english")
        bi_mat = bi_vec.fit_transform(corpus)
        bi_scores = bi_mat.mean(axis=0).A1
        bi_top = [
            bi_vec.get_feature_names_out()[i]
            for i in bi_scores.argsort()[::-1][:top_n]
        ]

        rows.append({
            "Bank": bank,
            "Top Keywords": ", ".join(uni_top),
            "Top Phrases":  ", ".join(bi_top),
        })

    return pd.DataFrame(rows).set_index("Bank")


def extract_top_ngrams(corpus: list[str],
                        n: int = 2,
                        top_k: int = 15) -> pd.DataFrame:
    """
    Return the top-k n-grams from *corpus* as a sorted DataFrame.
    """
    vec = CountVectorizer(ngram_range=(n, n), max_features=500,
                          stop_words="english")
    mat = vec.fit_transform(corpus)
    counts = mat.sum(axis=0).A1
    return (
        pd.DataFrame({"ngram": vec.get_feature_names_out(), "count": counts})
        .sort_values("count", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


# ============================================================================
# d)  LDA TOPIC MODELLING  (optional unsupervised discovery)
# ============================================================================

def run_lda(df: pd.DataFrame,
            text_col: str = "clean_text",
            n_topics: int = 6,
            top_words: int = 10,
            random_state: int = 42) -> tuple:
    """
    Fit an LDA model on the full corpus and return:
      (lda_model, vectorizer, topic_words_df, df_with_dominant_topic)

    *topic_words_df* — DataFrame with columns Topic_0 … Topic_N containing
                       the top words for each topic.
    *df_with_dominant_topic* — input df with a new 'lda_topic' column.
    """
    corpus = df[text_col].dropna().tolist()

    vec = CountVectorizer(max_features=1000, stop_words="english",
                          min_df=2, max_df=0.95)
    dtm = vec.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=20,
        learning_method="online",
    )
    lda.fit(dtm)

    feature_names = vec.get_feature_names_out()
    topic_words = {}
    for idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[::-1][:top_words]
        topic_words[f"Topic_{idx}"] = [feature_names[i] for i in top_idx]

    topic_words_df = pd.DataFrame(topic_words)

    # Assign dominant topic to each review
    dtm_full = vec.transform(df[text_col].fillna("").tolist())
    doc_topics = lda.transform(dtm_full)
    df_out = df.copy()
    df_out["lda_topic"] = doc_topics.argmax(axis=1)

    return lda, vec, topic_words_df, df_out


# ============================================================================
# VISUALISATION FUNCTIONS
# ============================================================================

def _theme_order(df: pd.DataFrame, theme_col: str) -> list[str]:
    """Return themes sorted by overall frequency (desc)."""
    return (
        df[theme_col]
        .str.split(", ")
        .explode()
        .value_counts()
        .index.tolist()
    )


def plot_themes_per_bank(df: pd.DataFrame,
                          theme_col: str = "themes",
                          bank_col: str = "bank",
                          figsize: tuple = (14, 6)) -> None:
    """
    Stacked bar chart: theme proportions per bank.
    """
    exploded = explode_themes(df, theme_col, bank_col)
    pivot = (
        pd.crosstab(exploded[bank_col], exploded[theme_col], normalize="index")
        .round(4)
    )
    # Consistent column order
    col_order = [c for c in THEME_COLORS if c in pivot.columns]
    pivot = pivot[col_order]

    colors = [THEME_COLORS.get(c, "#9E9E9E") for c in pivot.columns]

    ax = pivot.plot(kind="bar", stacked=True, figsize=figsize,
                    color=colors, edgecolor="white", linewidth=0.4)
    ax.set_title("Theme Distribution by Bank (Proportion)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Bank", fontsize=12)
    ax.set_ylabel("Proportion of Reviews", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Theme", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    plt.show()

    # Also print the numeric table
    print("\n=== Theme proportions per bank ===")
    print(pivot.to_string())


def plot_theme_sentiment(df: pd.DataFrame,
                          theme_col: str = "themes",
                          sentiment_col: str = "vader_compound",
                          bank_col: str = "bank",
                          figsize: tuple = (13, 6)) -> None:
    """
    Horizontal bar: mean VADER score per theme, faceted by bank.
    Positive (green) = driver; Negative (red) = pain point.
    """
    exploded = explode_themes(df, theme_col, bank_col)
    banks = sorted(exploded[bank_col].unique())
    n = len(banks)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]

    for ax, bank in zip(axes, banks):
        grp = exploded[exploded[bank_col] == bank]
        agg = (
            grp.groupby(theme_col)[sentiment_col]
            .mean()
            .sort_values()
        )
        colors = [
            THEME_COLORS.get(t, "#9E9E9E")
            for t in agg.index
        ]
        agg.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_title(f"{bank}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean VADER compound", fontsize=10)
        ax.set_ylabel("Theme" if ax is axes[0] else "")

    fig.suptitle("Theme Sentiment — Drivers (+) vs Pain Points (−)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_lda_topics(topic_words_df: pd.DataFrame,
                     lda_model,
                     top_words: int = 10,
                     figsize: tuple = (15, 8)) -> None:
    """
    Grid of bar charts showing word weights for each LDA topic.
    """
    n_topics = lda_model.n_components
    ncols = 3
    nrows = int(np.ceil(n_topics / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize, constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    palette = list(mcolors.TABLEAU_COLORS.values())

    for idx, ax in enumerate(axes_flat):
        if idx >= n_topics:
            ax.set_visible(False)
            continue
        col = f"Topic_{idx}"
        words = topic_words_df[col].tolist()[:top_words]
        weights = lda_model.components_[idx][
            [list(lda_model.components_[idx].argsort()[::-1]).index(i)
             for i in range(top_words)]
        ]
        # Recompute cleanly
        top_idx = lda_model.components_[idx].argsort()[::-1][:top_words]
        weights = lda_model.components_[idx][top_idx]

        ax.barh(words[::-1], weights[::-1],
                color=palette[idx % len(palette)], alpha=0.8)
        ax.set_title(f"LDA Topic {idx}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Word weight", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("LDA Unsupervised Topics — Top Words per Topic",
                 fontsize=14, fontweight="bold")
    plt.show()


def plot_keyword_heatmap(df: pd.DataFrame,
                          text_col: str = "clean_text",
                          bank_col: str = "bank",
                          top_n: int = 20,
                          figsize: tuple = (14, 7)) -> None:
    """
    Heatmap of mean TF-IDF scores for the top-N global keywords, broken down
    by bank.  Reveals which banks are associated with which keywords.
    """
    # Global vocabulary — top_n keywords across all banks combined
    global_vec = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2),
                                  stop_words="english")
    global_vec.fit(df[text_col].fillna(""))
    vocab = global_vec.get_feature_names_out()

    bank_scores = {}
    for bank, grp in df.groupby(bank_col):
        mat = global_vec.transform(grp[text_col].fillna(""))
        bank_scores[bank] = mat.mean(axis=0).A1

    heatmap_df = pd.DataFrame(bank_scores, index=vocab).T

    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_df,
        cmap="YlOrRd",
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Mean TF-IDF weight"},
    )
    plt.title(f"Top-{top_n} Keyword TF-IDF Weights by Bank",
              fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Keyword / Phrase", fontsize=11)
    plt.ylabel("Bank", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_wordcloud_per_bank(df: pd.DataFrame,
                             text_col: str = "clean_text",
                             bank_col: str = "bank",
                             figsize_per: tuple = (7, 4)) -> None:
    """
    One word-cloud per bank, rendered side-by-side.
    Requires the `wordcloud` package.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[WordCloud] Install with: pip install wordcloud")
        return

    banks = sorted(df[bank_col].unique())
    fig, axes = plt.subplots(
        1, len(banks),
        figsize=(figsize_per[0] * len(banks), figsize_per[1])
    )
    if len(banks) == 1:
        axes = [axes]

    colormaps = ["viridis", "plasma", "cividis", "magma"]
    for ax, bank, cmap in zip(axes, banks, colormaps):
        text = " ".join(df[df[bank_col] == bank][text_col].dropna())
        wc = WordCloud(
            width=700, height=380,
            background_color="white",
            colormap=cmap,
            max_words=120,
            collocations=False,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{bank}", fontsize=13, fontweight="bold", pad=8)

    fig.suptitle("Review Word Clouds by Bank",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_ngram_comparison(df: pd.DataFrame,
                           text_col: str = "clean_text",
                           bank_col: str = "bank",
                           n: int = 2,
                           top_k: int = 12,
                           figsize: tuple = (16, 5)) -> None:
    """
    Side-by-side horizontal bar charts of top n-grams for each bank.
    """
    banks = sorted(df[bank_col].unique())
    fig, axes = plt.subplots(1, len(banks), figsize=figsize)
    if len(banks) == 1:
        axes = [axes]

    palettes = ["Blues_r", "Oranges_r", "Greens_r"]
    label = "Bigrams" if n == 2 else ("Trigrams" if n == 3 else f"{n}-grams")

    for ax, bank, pal in zip(axes, banks, palettes):
        corpus = df[df[bank_col] == bank][text_col].dropna().tolist()
        ng_df = extract_top_ngrams(corpus, n=n, top_k=top_k)
        sns.barplot(data=ng_df, y="ngram", x="count", ax=ax, palette=pal)
        ax.set_title(f"{bank} — Top {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Count", fontsize=9)
        ax.set_ylabel("")

    plt.suptitle(f"Top {label} per Bank", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_theme_count_grid(df: pd.DataFrame,
                           theme_col: str = "themes",
                           bank_col: str = "bank",
                           figsize: tuple = (14, 5)) -> None:
    """
    Grouped bar chart: raw review counts per theme, faceted by bank.
    """
    exploded = explode_themes(df, theme_col, bank_col)
    pivot = pd.crosstab(exploded[theme_col], exploded[bank_col])
    col_order = [c for c in THEME_COLORS if c in pivot.index]
    if col_order:
        pivot = pivot.reindex(col_order)

    ax = pivot.plot(kind="bar", figsize=figsize,
                    color=[THEME_COLORS.get(b, "#888") for b in pivot.columns],
                    edgecolor="white", linewidth=0.4)
    ax.set_title("Review Count per Theme by Bank", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Theme", fontsize=12)
    ax.set_ylabel("Review Count", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(title="Bank")
    plt.tight_layout()
    plt.show()


# ============================================================================
# CONVENIENCE RUNNER
# ============================================================================

def run_full_thematic_pipeline(df: pd.DataFrame,
                                text_col: str = "clean_text",
                                review_col: str = "review",
                                bank_col: str = "bank",
                                sentiment_col: str = "vader_compound",
                                run_lda_flag: bool = True,
                                n_lda_topics: int = 6) -> pd.DataFrame:
    """
    End-to-end thematic analysis pipeline.

    Steps
    -----
    1. Assign supervised themes
    2. Print keyword extraction table
    3. All visualisations
    4. Optionally run LDA and visualise topics

    Returns the enriched DataFrame.
    """
    print("=" * 60)
    print("STEP 1 — Assigning supervised themes …")
    df = df.copy()
    df["themes"] = df[review_col].apply(assign_themes)
    print(f"  Theme column added. Sample:\n{df['themes'].value_counts().head(8).to_string()}")

    print("\nSTEP 2 — TF-IDF keyword extraction per bank …")
    kw_df = extract_keywords_tfidf(df, text_col=text_col, bank_col=bank_col)
    print(kw_df.to_string())

    print("\nSTEP 3 — N-gram comparison (bigrams) …")
    plot_ngram_comparison(df, text_col=text_col, bank_col=bank_col, n=2)

    print("\nSTEP 4 — Theme distribution per bank …")
    plot_themes_per_bank(df, theme_col="themes", bank_col=bank_col)

    print("\nSTEP 5 — Theme count grid …")
    plot_theme_count_grid(df, theme_col="themes", bank_col=bank_col)

    if sentiment_col in df.columns:
        print("\nSTEP 6 — Theme sentiment (drivers vs pain points) …")
        plot_theme_sentiment(df, theme_col="themes",
                              sentiment_col=sentiment_col, bank_col=bank_col)

    print("\nSTEP 7 — TF-IDF keyword heatmap …")
    plot_keyword_heatmap(df, text_col=text_col, bank_col=bank_col)

    print("\nSTEP 8 — Word clouds per bank …")
    plot_wordcloud_per_bank(df, text_col=text_col, bank_col=bank_col)

    if run_lda_flag:
        print(f"\nSTEP 9 — LDA topic modelling ({n_lda_topics} topics) …")
        lda_model, vec, topic_words_df, df = run_lda(
            df, text_col=text_col, n_topics=n_lda_topics
        )
        print("\n  LDA top words per topic:")
        print(topic_words_df.to_string())
        plot_lda_topics(topic_words_df, lda_model)

    print("\n" + "=" * 60)
    print("Thematic analysis complete ✓")
    return df