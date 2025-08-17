import re
import pandas as pd

def clean_text(t: str) -> str:
    t = re.sub(r"http\S+","", str(t))
    t = re.sub(r"\s+"," ", t).strip()
    return t

def build_summarization_pairs(df, group_col="video_id", max_comments=120, extract_k=5):
    pairs = []
    for vid, g in df.groupby(group_col):
        g = g.sort_values("likes", ascending=False).head(max_comments)
        doc = " ".join(clean_text(x) for x in g["text"].tolist())
        pseudo = " ".join(clean_text(x) for x in g["text"].head(extract_k).tolist())
        if len(doc) > 80 and len(pseudo) > 10:
            pairs.append({"doc": doc[:4000], "summary": pseudo[:512], "video_id": vid})
    return pd.DataFrame(pairs)
