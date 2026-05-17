# Ethiopian Fintech Review Analytics Pipeline

An automated data engineering and NLP pipeline designed to scrape, clean, and analyze customer feedback from major Ethiopian banking applications. This repository serves as a foundational ecosystem for identifying product friction points, system stability issues, and competitive performance gaps.

---

## Project Purpose and Business Context
As digital banking scales rapidly in East Africa, app store reviews represent a critical stream of raw user feedback. This project automates the aggregation and text processing of unstructured mobile app reviews to provide actionable telemetry for product management teams, specifically mapping feedback to business-relevant friction points like OTP delivery failures, network crashes, and transactional performance.

## Data Architecture and Methodology
The data processing lifecycle is modularized into three distinct stages:
1. **Extraction (Scraping):** Automated data extraction from the Google Play Store using the `google-play-scraper` engine, pulling metadata and historical strings natively.
2. **Preprocessing (NLP):** Pure Python and NLTK-driven cleaning pipeline that standardizes strings, tokenizes sentences, filters standard English stopwords, and lemmatizes words down to root concepts to enable structured text modeling.
3. **Analytics (Thematic and Sentiment):** Sentiment classification utilizing a serverless pre-trained DistilBERT transformer alongside vectorization-based keyword extraction (TF-IDF/Bag-of-Words) to map clusters into high-level business categories.

## Pipeline Scope and Specifications
* **Target Banks:** Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank.
* **Temporal Coverage (Date Range):** December 2025 to May 2026.
* **Dataset Volume:** 1,800 combined clean records aggregated across platforms (approximately 600 reviews per bank).

## Known Limitations
* **Language Bias:** The text classification models are optimized exclusively for English. Reviews written in Amharic, Oromo, or Latin-script transliterations (Ethio-English) are filtered out or underrepresented during NLP processing.
* **API Rate Limits:** The free Hugging Face serverless inference layer imposes strict concurrent request limits, requiring execution throttling when handling thousands of rows.

---

## Project Structure and Navigation

```text
├── data/
│   ├── raw/                # Unaltered, source CSV files directly from the scraper.
│   └── processed/          # Cleaned, tokenized, and sentiment-scored production tables.
├── notebooks/              # Jupyter workspaces for exploratory execution and visual modeling.
├── scripts/                # Production-ready executable Python components.
├── src/                    # Core modular architecture packages.
│   ├── __init__.py         # Initializes the src directory as a Python package.
│   ├── sentiment_analysis.py # Hugging Face serverless client initialization and sentiment scoring.
│   └── thematic_analysis.py  # Keyword grouping, TF-IDF weights, and plot distributions.
├── tests/                  # Automated validation unit tests.
│   └── test_analysis.py    # Unit tests for verification of data cleaning and theme mapping logic.
├── .env.example            # Template detailing required hidden variables (HF_TOKEN).
├── .github/
│   └── workflows/
│       └── unittest.yml    # CI/CD pipeline configuration for automated test runs on GitHub.
├── .gitignore              # Enforced files exclusion configurations (pycache, .env, venv).
└── requirements.txt        # Frozen dependency versioning lists.