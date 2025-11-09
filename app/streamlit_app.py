from __future__ import annotations

import os, re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# ================================
# page config
# ================================
st.set_page_config(
    page_title="TubePulse — YouTube Comment Sentiment & Summary",
    page_icon="📺",
    layout="wide",
)

# ---------------- small style touch (subtle and readable) ----------------
st.markdown(
    """
    <style>
      /* slightly tighter padding at the top so content appears sooner */
      .block-container {padding-top: 1.5rem;}
      /* simple card look for dataframes */
      .stDataFrame {border-radius: 14px; overflow: hidden; border: 1px solid #eaeaea;}
      /* keep metrics visually centered */
      .element-container:has(.metric-container) {text-align: center;}
      /* soften caption appearance */
      .stCaption {opacity: 0.85;}
      /* helper class for small notes */
      .note {font-size: 0.9rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# paths & constants
# ================================
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

SENTI_DIRS = [
    ROOT / "ml" / "models" / "sentiment_en",
    ROOT / "models" / "sentiment_en",
    Path.cwd() / "ml" / "models" / "sentiment_en",
    Path.cwd() / "models" / "sentiment_en",
]

BASE_SUMM = "sshleifer/distilbart-cnn-12-6"  # compact english summarizer chosen for speed and clarity

GEN_CFG = dict(  # decoding setup that balances detail and readability for summaries
    num_beams=4,
    max_length=160,
    min_length=int(160 * 0.6),
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    early_stopping=True,
)

# ================================
# utilities
# ================================
def get_api_key() -> str:
    """check env first, then streamlit secrets."""
    key = os.getenv("YOUTUBE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        return ""

def extract_video_id(url: str) -> str:
    """
    supports these common formats:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/shorts/VIDEOID
    """
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", url)
    if not m:
        raise ValueError("Could not parse a video ID from the URL.")
    return m.group(1)

# ================================
# caches: clients & models
# ================================
@st.cache_resource(show_spinner=False)
def yt_client(api_key: str):
    try:
        from googleapiclient.discovery import build
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "google-api-python-client is not installed. "
            "Install it with: pip install google-api-python-client"
        ) from e
    return build("youtube", "v3", developerKey=api_key)

@st.cache_data(show_spinner=True)
def fetch_top_level_comments(
    api_key: str,
    video_id: str,
    max_total: int = 2000,
    order: str = "relevance",
) -> pd.DataFrame:
    """
    fetch only top-level comments for a video.

    returns columns:
      platform, video_id, comment_id, parent_id (None), author, text,
      like_count, published_at, updated_at, source_url
    """
    youtube = yt_client(api_key)
    out, page, fetched = [], None, 0

    while True:
        req = youtube.commentThreads().list(
            part="snippet",            # replies are skipped on purpose so we stay at top-level
            videoId=video_id,
            maxResults=100,
            pageToken=page,
            textFormat="plainText",
            order=order,               # either "relevance" or "time"
        )
        res = req.execute()
        items = res.get("items", [])

        for it in items:
            top = it["snippet"]["topLevelComment"]
            sn  = top["snippet"]
            out.append({
                "platform": "youtube",
                "video_id": video_id,
                "comment_id": top["id"],
                "parent_id": None,
                "author": sn.get("authorDisplayName",""),
                "text": str(sn.get("textOriginal") or sn.get("textDisplay","")),
                "like_count": int(sn.get("likeCount",0) or 0),
                "published_at": sn.get("publishedAt",""),
                "updated_at": sn.get("updatedAt",""),
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
            })
            fetched += 1
            if fetched >= max_total:
                break

        if fetched >= max_total:
            break
        page = res.get("nextPageToken")
        if not page:
            break

    df = pd.DataFrame(out)
    if not df.empty:
        # light clean: collapse whitespace, keep one row per comment id
        df["text"] = df["text"].astype(str).str.replace(r"\s+"," ", regex=True).str.strip()
        df = df.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    return df

def first_existing_dir(candidates: list[Path]) -> Optional[str]:
    """return the first existing directory path from a list, or None if none exist."""
    for p in candidates:
        if p.is_dir():
            return str(p)
    return None

@st.cache_resource(show_spinner=False)
def load_sentiment():
    """
    prefer a local fine-tuned model if present,
    otherwise fall back to a public model and adapt its output to a simple view.
    """
    local_path = first_existing_dir(SENTI_DIRS)
    if local_path:
        tok = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path).eval()
        id2label = model.config.id2label
        mode = "local"
    else:
        # fallback: 3-class roberta model; we later map it into (neg, pos)
        fallback_hf = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tok = AutoTokenizer.from_pretrained(fallback_hf)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_hf).eval()
        id2label = model.config.id2label
        mode = "fallback"
    return tok, model, id2label, mode

@st.cache_resource(show_spinner=False)
def load_base_summarizer():
    """load the summarizer used for turning comments into a short overview."""
    tok = AutoTokenizer.from_pretrained(BASE_SUMM)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_SUMM).eval()
    return tok, model

# ================================
# inference helpers
# ================================
def predict_sentiment(
    texts: List[str],
    tok, model, id2label, mode: str,
    max_len: int = 160,
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:
    """
    run batched sentiment inference and return labels and probabilities.
    if using the fallback model, neutral is merged away so we keep a simple binary view.
    """
    labels_out, probs_out = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        if mode == "fallback" and probs.shape[1] == 3:
            # assume [neg, neu, pos] and keep only neg/pos columns
            probs = probs[:, [0, 2]]
        ids = probs.argmax(-1)
        labels = [id2label.get(int(i), str(i)) for i in ids]
        labels_out.extend(labels)
        probs_out.extend(probs)
    return labels_out, np.array(probs_out)

def summarize_texts(
    texts: List[str],
    tok, model,
    top_n_doc: int = 120,
    max_in: int = 512,
    gen_cfg: dict = GEN_CFG,
) -> str:
    """
    join the top n comments into one block and generate a short summary from it.
    """
    if not texts:
        return ""
    doc = " ".join(texts[:top_n_doc])
    enc = tok([doc], truncation=True, padding=True, max_length=max_in, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, **gen_cfg)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

# ================================
# ui — TubePulse
# ================================
st.markdown("## 📺 TubePulse — *Understand the vibe of any YouTube video in seconds*")
st.caption("Paste a YouTube link. We grab **top-level comments**, score **sentiment with confidence**, and craft a **concise summary** for quick takeaways.")

with st.sidebar:
    st.header("⚙️ Options")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("🔑 YouTube API key", type="password", help="Not stored; used for this session only.")
    max_comments = st.slider("🗂️ Max comments", 0, 1000, 100, step=10)
    order = st.selectbox("🧮 Fetch order", ["relevance", "time"])
    top_n_doc = st.slider("📝 Summary uses top N comments", 0, 1000, 100, step=10)
    st.divider()
    st.markdown("<span class='note'>Tip: Higher N yields richer summaries but takes longer.</span>", unsafe_allow_html=True)

url = st.text_input("🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
go = st.button("🔎 Analyze", type="primary")

if go:
    # validate url and extract id
    try:
        vid = extract_video_id(url.strip())
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    if not api_key:
        st.error("❌ Missing YouTube API key. Set env `YOUTUBE_API_KEY`, add to `.streamlit/secrets.toml`, or enter it in the sidebar.")
        st.stop()

    # fetch comments from youtube api
    with st.spinner("📥 Fetching top-level comments…"):
        try:
            df = fetch_top_level_comments(api_key, vid, max_total=max_comments, order=order)
        except Exception as e:
            st.exception(e)
            st.stop()

    if df.empty:
        st.warning("⚠️ No comments found (or API quota reached). Try a different video.")
        st.stop()

    # sort to surface higher-signal comments first
    df = df.sort_values("like_count", ascending=False).reset_index(drop=True)

    # load summarizer and sentiment models
    tok_s, model_s = load_base_summarizer()
    tok_c, model_c, id2label, mode = load_sentiment()

    # run sentiment prediction
    with st.spinner("🧠 Scoring sentiment…"):
        labels, probs = predict_sentiment(df["text"].tolist(), tok_c, model_c, id2label, mode)
    df["sentiment"] = labels
    df["p_neg"] = probs[:, 0]
    df["p_pos"] = probs[:, -1]  # last column is treated as positive side in our view

    # build summary from comments
    with st.spinner("📝 Generating summary…"):
        summary = summarize_texts(df["text"].tolist(), tok_s, model_s, top_n_doc=top_n_doc)

    # ========================
    # results
    # ========================
    colA, colB, colC = st.columns(3)
    colA.metric("💬 Comments analyzed", f"{len(df):,}")
    colB.metric("🧵 Top-level only", "Yes")
    colC.metric("⏱️ Order", order.capitalize())

    st.subheader("📝 Summary of comments")
    st.write(summary)

    st.subheader("📊 Sentiment snapshot")
    counts = df["sentiment"].value_counts().reindex(["negative", "positive"]).fillna(0).astype(int)
    st.bar_chart(counts)

    st.subheader("🏆 Top comments by confidence")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Most positive**")
        st.dataframe(
            df.sort_values("p_pos", ascending=False)[["author","text","like_count","p_pos"]].head(5),
            use_container_width=True
        )
    with c2:
        st.markdown("**🚫 Most negative**")
        st.dataframe(
            df.sort_values("p_neg", ascending=False)[["author","text","like_count","p_neg"]].head(5),
            use_container_width=True
        )

    st.subheader("⬇️ Download")
    st.download_button(
        "Download scored comments (CSV)",
        data=df[["author","text","like_count","sentiment","p_neg","p_pos","published_at","source_url"]].to_csv(index=False).encode("utf-8"),
        file_name=f"tubepulse_{vid}.csv",
        mime="text/csv",
    )

# ================================
# about
# ================================
with st.expander("ℹ️ About TubePulse"):
    st.markdown(
        """
**TubePulse** extracts **top-level YouTube comments**, computes **sentiment with confidence**, and produces a **concise summary**—so you can grasp the vibe of a video at a glance.

**How it works**
- **Data**: YouTube Data API v3 (top-level comments only to avoid reply-thread bias).
- **Sentiment**: Uses your local fine-tuned binary model if found at `models/sentiment_en/`.  
  Otherwise falls back to a strong public RoBERTa sentiment model and maps to *(neg, pos)*.
- **Summarization**: Base **DistilBART** (`sshleifer/distilbart-cnn-12-6`) with tuned decoding
  *(4-beam search, length penalty, no-repeat n-grams)* for clean, readable summaries.
- **Ranking**: Surfaces the most positive/negative comments by **model confidence** and includes raw
  probabilities in the export.

**Why only top-level?**  
Top-level comments better reflect the **overall audience reaction**. Replies are often side-threads or debates
that can skew the distribution.

**Reproducibility**
- Provide `YOUTUBE_API_KEY` via environment or `.streamlit/secrets.toml`.
- Drop your own model into `models/sentiment_en/` to override the fallback—no code changes needed.
        """
    )
