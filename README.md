# 🏨 HotelSense AI

### An Explainable Multimodal Hotel Guest Satisfaction Prediction & Review Intelligence Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/HotelSense_AI.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Google%20Colab-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> A complete, production-style ML system in a **single, self-contained Colab notebook**. It predicts hotel guest satisfaction from **both structured review metadata and natural-language review text**, and turns raw reviews into actionable **review intelligence** — aspect sentiment, topic discovery, semantic search, summarization, and a retrieval-augmented Q&A assistant.

---

## ✨ Highlights

- **Runs top-to-bottom in Colab** with one click — no manual downloads, uploads, or file creation.
- **Self-healing data ingestion.** Tries HuggingFace → CSV mirrors → a built-in aspect-aware **synthetic generator**, so the notebook *always* produces a working dataset, even with no network access.
- **Dual modelling objective:** 3-class satisfaction **classification** *and* review-score **regression**, trained and compared side-by-side.
- **Multimodal:** fuses structured features with sentence-transformer embeddings in a hybrid neural network.
- **Explainable:** SHAP global + local explanations, plus 16-aspect sentiment analysis.
- **Review intelligence:** BERTopic complaint discovery, transformer summarization, FAISS semantic search, and a RAG assistant.
- **Interactive:** a polished 5-tab **Gradio** app.
- **Reproducible & persisted:** fixed seeds, cached embeddings, and every artifact saved to disk, plus an auto-generated **HTML/PDF report**.

---

## 🚀 Quickstart

### Option A — Google Colab (recommended)
1. Click the **Open in Colab** badge above (after pushing to your repo).
2. `Runtime → Run all`.
3. Wait for the pipeline to run — on first run it installs dependencies and downloads a few small open-source models. A public **Gradio link** appears near the end.

### Option B — Local Jupyter
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt   # optional; the notebook also self-installs
jupyter notebook HotelSense_AI.ipynb
```
A GPU is optional — the notebook auto-detects it and falls back to CPU cleanly.

---

## ⚙️ Configuration

Every knob lives in a single `Config` dataclass near the top of the notebook. The most useful:

| Setting | Default | What it does |
|---|---|---|
| `sample_size` | `15000` | Rows kept after loading — the master runtime dial. Lower for a fast demo, raise for a fuller run. |
| `optuna_trials` | `20` | Number of Optuna hyperparameter-search trials. |
| `transformer_embeddings_enabled` | `True` | Use sentence-transformer embeddings (vs. a TF-IDF/SVD fallback). |
| `bertopic_enabled` | `True` | Run BERTopic topic modeling. |
| `summarization_enabled` | `True` | Load the transformer summarizer. |
| `rag_enabled` | `True` | Enable the RAG assistant (cross-encoder rerank + generator). |
| `gradio_enabled` | `True` | Build and launch the Gradio app. |

---

## 📓 What the notebook does (section by section)

1. **Environment Setup** — hardware detection + dependency installation.
2. **Configuration** — `Config` dataclass, fixed seeds, directory tree, logging.
3. **Automated Data Acquisition** — multi-source loader with synthetic fallback.
4. **Validation & Cleaning** — schema/type/range checks, deduplication, placeholder removal.
5. **Target Construction** — 3-class satisfaction label + review-score regression target.
6. **Exploratory Data Analysis** — distributions, word clouds, n-grams, correlations, PCA/t-SNE/UMAP.
7. **NLP Pipeline** — cleaning, lemmatization, readability, POS/NER, VADER + transformer sentiment/emotion.
8. **Aspect-Based Sentiment Analysis** — 16 hotel aspects scored positive / neutral / negative.
9. **Feature Engineering** — temporal features, encodings, correlation-based pruning.
10. **Embeddings** — TF-IDF + cached sentence-transformer vectors.
11. **Model Zoo** — Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, MLP, and a hybrid multimodal neural net — for **both** classification and regression.
12. **Hyperparameter Optimization** — Optuna with a median pruner.
13. **Evaluation** — confusion matrix, ROC/PR/calibration curves, residual & error plots.
14. **Explainability** — SHAP global (beeswarm + bar) and local (waterfall) explanations.
15. **Topic Modeling** — BERTopic complaint discovery (KMeans fallback).
16. **Summarization** — open-source abstractive summarizer with an extractive fallback.
17. **Vector DB + Semantic Search** — persistent FAISS index over review embeddings.
18. **RAG Assistant** — retrieve → cross-encoder rerank → open-source LLM generation.
19. **Gradio App** — 5-tab interface (see below).
20. **Persistence & Reporting** — save all artifacts + generate an HTML/PDF report.

---

## 🖥️ Gradio App Tabs

1. **Satisfaction Prediction** — structured inputs + review text → predicted class, score, confidence, and aspect drivers.
2. **Semantic Review Search** — natural-language search over the review corpus.
3. **Analytics Dashboard** — interactive Plotly charts (scores by trip type, aspect sentiment, temporal trends).
4. **Review Summarization** — per-hotel executive summaries.
5. **AI RAG Assistant** — grounded natural-language Q&A over all reviews.

---

## 🧰 Tech Stack

**ML / data:** scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, Optuna, SHAP
**NLP:** Hugging Face Transformers, Sentence-Transformers, spaCy, NLTK, VADER, TextBlob, textstat, BERTopic
**Vector search:** FAISS
**App / viz:** Gradio, Matplotlib, Seaborn, Plotly, WordCloud

Default open-source models: `all-MiniLM-L6-v2` (embeddings), `ms-marco-MiniLM-L-6-v2` (reranker), `distilbart-cnn-12-6` (summarization), `flan-t5-base` (RAG generation).

---

## 📊 About the Data

The pipeline targets a Booking.com-style review schema (`hotel_name`, `hotel_location`, `positive_review`, `negative_review`, `review_score`, `trip_type`, `room_type`, `nationality`, `stay_duration`, `review_date`, …).

If no public source is reachable, a **synthetic generator** produces a realistic, aspect-aware dataset so the whole pipeline still runs and every subsystem is demonstrable. Metrics on the synthetic data are intentionally realistic (not perfect) — for example ~0.61 macro-F1 for the 3-class task and ~0.5 R² for score regression — because the generator injects genuine noise. Swap in the real corpus (e.g. via a Kaggle API key) and raise `sample_size` for a full-scale run; everything downstream works identically.

---

## 📁 Repository Layout

```
.
├── HotelSense_AI.ipynb     # the entire project — one notebook
├── README.md
├── requirements.txt        # optional; the notebook self-installs
└── .gitignore
```

At runtime the notebook creates a `HotelSenseAI/` tree (data, models, embeddings, faiss, outputs, logs) — **this is generated and should not be committed** (see `.gitignore` below).

### Suggested `.gitignore`
```gitignore
HotelSenseAI/
*.joblib
*.pt
*.npy
*.faiss
__pycache__/
.ipynb_checkpoints/
```

---

## ⚠️ Notes & Limitations

- First run downloads several hundred MB of models and installs packages — subsequent runs reuse cached embeddings and the FAISS index.
- The Gradio app launches with a temporary public share link (valid ~72h) for convenience.
- Heavy stages (transformer scoring, UMAP, BERTopic, Optuna) are individually toggleable in `Config` if you need a faster run.
- This is a demonstration/portfolio-grade pipeline, not a hardened production service — see the "Future Improvements" section of the generated report for a roadmap (domain-tuned ABSA, multilingual support, a low-latency FAISS API, drift monitoring, etc.).

---

## 📄 License

Released under the MIT License. See `LICENSE` for details.

---

<p align="center"><i>Built as an end-to-end, explainable, multimodal review-intelligence platform.</i> 🏨</p>
