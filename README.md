# 📺 TubePulse — YouTube Comment Sentiment & Summary

A lightweight, end-to-end **YouTube comment analysis** app built with Streamlit.

TubePulse is designed for quick insights: fetch **top-level comments** from any YouTube video, score **sentiment with confidence**, and generate a **concise, readable summary**—ready for export and downstream analysis.

👉 **Live demo:** [TubePulse App](https://yt-comment-analyzer-8upz5k8zd7hkxtwotjm45d.streamlit.app/)

---

## 🌐 Overview

TubePulse provides a clean workflow:

1. **Paste URL** – We extract the video ID and fetch **top-level** comments via the YouTube Data API (replies are excluded to avoid thread bias).
2. **Analyze** – Run sentiment on each comment with **probabilities** and produce an abstractive **summary** of the discussion.
3. **Review & Export** – Inspect most positive/negative comments by confidence and **download a scored CSV**.

Useful for:

* Rapid audience-reaction scans (creator teams, comms, marketing)
* Baseline moderation and triage
* Research and experimentation with comment-level NLP

---

## 🎯 Core Design Objectives

1. **Simplicity & Speed**

   * One input (YouTube URL) → one click → insights.
   * Sensible defaults; works out-of-the-box.

2. **Signal over Noise**

   * **Top-level comments only** to reflect overall reception.
   * Optional sort by likes (highest-signal comments inform the summary).

3. **Transparent Confidence**

   * Sentiment includes **per-class probabilities** for ranking and QA.
   * Export preserves raw scores for reproducible analysis.

4. **Model Flexibility**

   * **Local fine-tuned** binary model supported (drop into `ml/models/sentiment_en/`).
   * **Robust fallback** to a high-quality public model if no local model is found.
   * Summarization uses a strong, compact **base** encoder-decoder (no extra adapters needed).

---

## 🧠 Method: What’s Under the Hood?

**Data ingestion**

* YouTube Data API v3 → **commentThreads** (top-level only), with optional ordering by **relevance** or **time**.

**Sentiment**

* Prefers your **local fine-tuned** binary model at `ml/models/sentiment_en/`.
* If unavailable, falls back to **`cardiffnlp/twitter-roberta-base-sentiment-latest`** and maps 3-class outputs to **(negative, positive)** (neutral dropped), preserving confidences.

**Summarization**

* Ships **`sshleifer/distilbart-cnn-12-6`** (English) with tuned decoding:

  * `num_beams=4`, `no_repeat_ngram_size=3`, `length_penalty=2.0`,
  * `min_length≈60%` of `max_length` for readability/coverage.

**Outputs**

* Distribution chart (negative/positive)
* **Top positive/negative** comments by model confidence
* Summary text
* **CSV export** including `p_neg`, `p_pos`, and metadata columns

---

## 🔧 Functional Breakdown

### 1) Input & Fetch

**Inputs**

* YouTube URL
* API key (via `.env` or Streamlit secrets)
* Options: **Max comments**, fetch **order** (relevance/time), **N** comments used for summary

**Process**

* Extract `video_id` from the URL.
* Fetch **top-level** comments only (replies excluded).
* Clean + de-duplicate, then sort by `like_count` for stronger signal.

---

### 2) Sentiment & Summary

**Sentiment**

* Tokenize comments → model inference → `sentiment` + `p_neg`/`p_pos`.
* Show distribution + confidence-ranked examples.

**Summary**

* Concatenate the top-liked comments (configurable `top_n`) and run abstractive summarization to produce a single, concise summary.

---

### 3) Export & Review

* **Download CSV** with: `author, text, like_count, sentiment, p_neg, p_pos, published_at, source_url`.
* Inspect most positive/negative comments (by confidence) in the app.

---

## 📦 Dependencies

```txt
streamlit
google-api-python-client
python-dotenv
pandas
numpy
torch
transformers
accelerate
evaluate
scikit-learn
```

*(Install via `pip install -r requirements.txt`.)*

---

## ▶️ Run Locally

```bash
# 1) Clone & enter
git clone https://github.com/<you>/yt-comment-analyzer.git
cd yt-comment-analyzer

# 2) (Optional) Create a venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install
pip install -r requirements.txt

# 4) Provide your YouTube API key (choose one)
#   A) .env (local; not committed)
#      Create ./.env with:
#      YOUTUBE_API_KEY=YOUR_KEY_HERE
#   B) Streamlit secrets (also not committed)
#      Create ./.streamlit/secrets.toml with:
#      YOUTUBE_API_KEY = "YOUR_KEY_HERE"

# 5) Run
streamlit run app/streamlit_app.py
```

Then open the local URL printed in your terminal.

---

## ⚠️ Notes & Limits

* **API quotas** apply (YouTube Data API v3). Heavy use may exhaust daily quota.
* **Language**: The default models are English-centric. If a video’s comments are mostly non-English, consider adding language filtering/translation before analysis.
* **Fallback**: If no local sentiment model is found, we use a robust public model and map 3-class → binary; expect slight differences in distribution.

---

## 🛣️ Roadmap

* Emotion / toxicity / topic clustering
* Optional replies analysis (thread-aware)
* Language detection and translation pipeline
* Per-author & temporal sentiment trends

---

## 🙏 Acknowledgments

* Sentiment fallback: **cardiffnlp/twitter-roberta-base-sentiment-latest**
* Summarization base: **sshleifer/distilbart-cnn-12-6**
* Built with **Streamlit**, **Transformers**, **PyTorch**, and **YouTube Data API v3**
