from googleapiclient.discovery import build
import re

def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", url)
    if not m:
        raise ValueError("Could not parse a video ID from the URL.")
    return m.group(1)

def get_yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)

def fetch_comments_all(youtube, video_id: str, max_total: int = 5000):
    out, page = [], None
    while True:
        req = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=page,
            textFormat="plainText"
        )
        res = req.execute()
        for it in res.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "author": top.get("authorDisplayName",""),
                "text": top.get("textDisplay",""),
                "likes": top.get("likeCount",0),
                "ts": top.get("publishedAt",""),
                "parent": None
            })
            for rep in it.get("replies", {}).get("comments", []):
                s = rep["snippet"]
                out.append({
                    "author": s.get("authorDisplayName",""),
                    "text": s.get("textDisplay",""),
                    "likes": s.get("likeCount",0),
                    "ts": s.get("publishedAt",""),
                    "parent": top.get("textDisplay","")
                })
        page = res.get("nextPageToken")
        if not page or len(out) >= max_total:
            break
    return out
