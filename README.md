# 📱 ZaloPay Review Severity Classification with Deep Learning

Automatically classifying the **severity** of customer feedback (not just sentiment) for Vietnam's leading e-wallet, ZaloPay — built to help fintech teams triage financial-risk complaints before they escalate.

> 🎓 Course project — Web Data Analytics, University of Economics and Law (UEL), VNU-HCMC (06/2026)
> 👥 Team of 3 · [My contribution](#-my-contribution): data pipeline, EDA, and PhoBERT fine-tuning

---

## 🎯 Why this project?

Most review-analysis systems stop at **sentiment** (positive / negative). But two 1-star reviews can mean very different things:

| Review | Sentiment | Real severity |
|---|---|---|
| "App hơi lag, load chậm" (app is a bit laggy) | Negative | Low — UX annoyance |
| "Giao dịch bị trừ tiền nhưng chưa được hoàn trả" (charged but never refunded) | Negative | **Critical** — money at risk |

Rating alone can't tell these apart either — some 5-star reviews still contain hidden complaints ("khen kèm chê"). This project builds a model that reads the **text itself** to flag the reviews a fintech company should act on first.

---

## 📊 Dataset

- **~74,600 reviews** scraped from the ZaloPay app on Google Play Store (`google-play-scraper`), covering 06/2025 – 06/2026
- Fields: rating, review text, timestamp, thumbs-up count, app version, reply status
- Severity labels derived from rating as an initial proxy, refined during preprocessing:

| Rating | Severity Label |
|---|---|
| 4–5 ★ | 0 – Normal |
| 3 ★ | 1 – Minor |
| 2 ★ | 2 – Serious |
| 1 ★ | 3 – Critical |

**Key challenge:** the dataset is heavily imbalanced — over 83% of reviews are 5-star, while the "critical" financial/security complaints that matter most make up a small minority.

---

## 🔧 Pipeline (KDD + Web Mining)

```
Raw reviews (Google Play)
        │
        ▼
Data Cleaning        →  dedup, missing values, feature engineering (review length, hour, weekday)
        │
        ▼
Text Cleaning        →  Unicode normalization, teencode/emoji normalization,
        │                Vietnamese word segmentation (underthesea)
        ▼
Class Balancing       →  Resampling (under/oversampling) + Class Weights
        │
        ▼
   ┌────────────┴────────────┐
   ▼                         ▼
TF-IDF + Logistic       PhoBERT (vinai/phobert-base-v2)
Regression (baseline)    fine-tuned end-to-end (PyTorch + HuggingFace)
   └────────────┬────────────┘
                ▼
     Severity prediction {0, 1, 2, 3}
                ▼
        Dashboard (insights for the business)
```

---

## 🤖 Models compared

| Metric | TF-IDF + Logistic Regression | PhoBERT Fine-tuning |
|---|---|---|
| Accuracy | 79% | **87%** |
| Macro F1-score | 0.79 | **0.87** |
| Precision — "Critical" class | 0.74 | **0.97** |
| Recall — "Critical" class | 0.76 | 0.78 |

**Why PhoBERT wins:** TF-IDF treats text as a bag of independent words, so it gets fooled by negation, sarcasm, and mixed-sentiment reviews (*"App tốt nhưng bị trừ tiền oan"* — "app is good but wrongly charged me"). PhoBERT's self-attention captures that context, which is exactly where the financial-risk signal lives.

At **97% precision** on the Critical class, a "Critical" prediction can be trusted almost every time — minimizing false alarms for the operations team.

---

## 💡 Key business insights

From analyzing the "Serious" + "Critical" reviews:

- **38%** — Transaction / transfer errors (money deducted, transfer stuck, refund delays)
- **24%** — Payment & bank-linking failures
- **20%** — App performance & stability (crashes, lag)
- **18%** — Account verification & OTP/KYC issues

👉 Full write-up, confusion matrices, word clouds, and the interactive dashboard are in [`/report`](./report) and [`/notebooks`](./notebooks).

---

## 🛠️ Tech stack

`Python` `Pandas` `NumPy` `google-play-scraper` `underthesea` (Vietnamese NLP) `Scikit-learn` `PyTorch` `HuggingFace Transformers` `PhoBERT` `Matplotlib` `Seaborn` `WordCloud`

---

## 📁 Repository structure

```
├── data/                 # raw & processed datasets (or scripts to fetch them)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_tfidf_logreg.ipynb
│   └── 05_phobert_finetuning.ipynb
├── dashboard/            # dashboard source
├── report/               # full written report (PDF)
└── README.md
```

## ▶️ How to run

```bash
git clone https://github.com/<username>/zalopay-severity-classification.git
cd zalopay-severity-classification
pip install -r requirements.txt
jupyter notebook notebooks/01_data_collection.ipynb
```

---

## 👤 My contribution

- Built the automated data-collection pipeline (`google-play-scraper`), handling pagination and rate-limiting to gather ~74,600 reviews
- Led exploratory data analysis (rating distribution, review-length patterns, word clouds) to surface the class-imbalance and mixed-sentiment challenges
- Implemented Vietnamese text preprocessing (Unicode/teencode normalization, `underthesea` tokenization) and the class-balancing strategy
- Fine-tuned PhoBERT end-to-end and benchmarked it against the TF-IDF baseline

**Trần Thị Kim Chi** · [LinkedIn](https://linkedin.com/in/kimchitran243) · kiimchitran2403@gmail.com
