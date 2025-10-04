# flash_index_app.py
import streamlit as st
import requests
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from langdetect import detect
from textblob import TextBlob
import pandas as pd
import time
import re
import math
import matplotlib.pyplot as plt
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG - Streamlit secrets
# You MUST add the following keys into Streamlit secrets:
# GOOGLE_API_KEY_FLASH_P
# GOOGLE_SEARCH_ENGINE_ID_FLASH_P
# GOOGLE_API_KEY_FLASH_N
# GOOGLE_SEARCH_ENGINE_ID_FLASH_N
# ---------------------------

# Page config should be before other Streamlit calls
st.set_page_config(page_title="Flash-P & Flash-N Index Creator", layout="wide")

API_KEY_P = st.secrets.get("GOOGLE_API_KEY_FLASH_P")
CX_P = st.secrets.get("GOOGLE_SEARCH_ENGINE_ID_FLASH_P")
API_KEY_N = st.secrets.get("GOOGLE_API_KEY_FLASH_N")
CX_N = st.secrets.get("GOOGLE_SEARCH_ENGINE_ID_FLASH_N")

if not all([API_KEY_P, CX_P, API_KEY_N, CX_N]):
    st.error(
        "Please add the four required secrets: "
        "GOOGLE_API_KEY_FLASH_P, GOOGLE_SEARCH_ENGINE_ID_FLASH_P, "
        "GOOGLE_API_KEY_FLASH_N, GOOGLE_SEARCH_ENGINE_ID_FLASH_N"
    )
    st.stop()

st.title("⚡ Flash-P & Flash-N — Dynamic Sentiment Index Creator")
st.markdown(
    """
Upload a CSV with headers: `symbol, Name, Weights`.  
This tool will search news using two Google Custom Search setups (one for Flash-P and one for Flash-N),
classify sentiment automatically, assign auto-weights, normalize the index and produce two CSVs:
- `flash_p_index.csv` (positive index)
- `flash_n_index.csv` (negative index)
"""
)

# ---------------------------
# Controls
# ---------------------------
uploaded = st.file_uploader("Upload CSV (symbol, Name, Weights)", type=["csv"])
num_results_per_symbol = st.number_input(
    "Results per symbol to fetch (per engine)", min_value=1, max_value=10, value=5, step=1
)
sleep_between_requests = st.number_input(
    "Sleep between web requests (seconds)", min_value=0.0, max_value=5.0, value=0.8, step=0.1
)

# ---------------------------
# Helpers
# ---------------------------
def normalize_header(h: str) -> str:
    return re.sub(r'\s+', '', h.strip().lower())

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_article_text(url: str, timeout: int = 8) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FlashIndexBot/1.0; +https://example.com)"
    }
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200 or not r.text:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        text = preprocess_text(text)
        # require some reasonable length
        if len(text.split()) < 40:
            return ""
        return text
    except Exception:
        return ""

# create and reuse google customsearch service objects
def make_service(api_key):
    try:
        return build("customsearch", "v1", developerKey=api_key)
    except Exception:
        return None

SERVICE_P = make_service(API_KEY_P)
SERVICE_N = make_service(API_KEY_N)

# cache search results to avoid repeated hits during debugging
@st.cache_data(show_spinner=False)
def google_search_cached(api_key_label: str, cx: str, query: str, num: int = 5):
    """
    api_key_label: "P" or "N" (used to pick the right service)
    """
    try:
        service = SERVICE_P if api_key_label == "P" else SERVICE_N
        if service is None:
            return []
        # The API allows up to 10 results per call
        res = service.cse().list(q=query, cx=cx, num=num).execute()
        items = res.get("items", []) or []
        results = []
        for it in items:
            results.append({
                "title": it.get("title"),
                "link": it.get("link"),
                "snippet": it.get("snippet", "")
            })
        return results
    except Exception as e:
        # cache empty result for this query
        return []

def sentiment_score(text: str) -> float:
    try:
        tb = TextBlob(text)
        return tb.sentiment.polarity
    except Exception:
        return 0.0

def compute_raw_weight(base_weight: float, avg_polarity: float, index_type: str = "P") -> float:
    p = max(-1.0, min(1.0, float(avg_polarity)))
    if index_type == "P":
        raw = base_weight * (1.0 + max(0.0, p))
    else:
        raw = base_weight * (1.0 + max(0.0, -p))
    return max(1e-8, raw)

def normalize_weights(df: pd.DataFrame, weight_col: str = "auto_weight") -> pd.DataFrame:
    total = df[weight_col].sum()
    if total <= 0 or math.isclose(total, 0.0):
        df["normalized_weight"] = 1.0 / len(df)
    else:
        df["normalized_weight"] = df[weight_col] / total
    return df

# ---------------------------
# Main
# ---------------------------
if uploaded is not None:
    try:
        df_input = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    header_map = {normalize_header(h): h for h in df_input.columns}
    if not all(k in header_map for k in ["symbol", "name", "weights"]):
        st.error("CSV must contain headers: symbol, Name, Weights (case-insensitive).")
        st.stop()

    df_input = df_input.rename(columns={
        header_map["symbol"]: "symbol",
        header_map["name"]: "name",
        header_map["weights"]: "base_weight"
    })
    df_input["base_weight"] = pd.to_numeric(df_input["base_weight"], errors="coerce").fillna(0.0)
    st.success(f"Loaded {len(df_input)} symbols.")
    st.subheader("Input Preview")
    st.dataframe(df_input.head(50))

    if st.button("▶️ Build Flash-P & Flash-N Indexes"):
        # prevent UI blocking by using spinner
        with st.spinner("Building indexes — running searches and sentiment analysis..."):
            # progress bar: use percent 0..100
            progress = st.progress(0)
            total_steps = max(1, len(df_input) * 2)
            step = 0

            rows_p = []
            rows_n = []

            # quick visible log
            log_container = st.empty()
            success_count = 0

            for i, row in df_input.iterrows():
                try:
                    symbol = str(row["symbol"]).strip()
                    name = str(row["name"]).strip()
                    base_weight = float(row["base_weight"])

                    query = f"{name} {symbol} news"

                    # Flash-P (engine P)
                    search_results_p = google_search_cached("P", CX_P, query, num=int(num_results_per_symbol))
                    sentiments_p = []
                    scraped_count_p = 0
                    for res in search_results_p:
                        url = res.get("link", "")
                        # respect sleep
                        time.sleep(float(sleep_between_requests))
                        text = ""
                        if url:
                            text = fetch_article_text(url)
                        if not text:
                            # fallback to snippet (short but better than nothing)
                            text = preprocess_text(res.get("snippet", "") or "")
                        if text:
                            scraped_count_p += 1
                            try:
                                # language detection on 1st 200 chars - if it errors, ignore
                                lang = detect(text[:200])
                                # proceed regardless of language (TextBlob may still give rough polarity)
                            except Exception:
                                pass
                            s = sentiment_score(text)
                            sentiments_p.append(s)

                    avg_polarity = (sum(sentiments_p) / len(sentiments_p)) if sentiments_p else 0.0
                    auto_weight_p = compute_raw_weight(base_weight, avg_polarity, index_type="P")
                    rows_p.append({
                        "symbol": symbol,
                        "name": name,
                        "base_weight": base_weight,
                        "num_articles": len(sentiments_p),
                        "avg_polarity": avg_polarity,
                        "auto_weight": auto_weight_p
                    })
                    step += 1
                    progress.progress(int(step / total_steps * 100))
                    log_container.info(f"[{i+1}/{len(df_input)}] {symbol} — Flash-P: {len(sentiments_p)} articles, avg_pol={avg_polarity:.4f}")

                    # Flash-N (engine N)
                    search_results_n = google_search_cached("N", CX_N, query, num=int(num_results_per_symbol))
                    sentiments_n = []
                    scraped_count_n = 0
                    for res in search_results_n:
                        url = res.get("link", "")
                        time.sleep(float(sleep_between_requests))
                        text = ""
                        if url:
                            text = fetch_article_text(url)
                        if not text:
                            text = preprocess_text(res.get("snippet", "") or "")
                        if text:
                            scraped_count_n += 1
                            try:
                                lang = detect(text[:200])
                            except Exception:
                                pass
                            s = sentiment_score(text)
                            sentiments_n.append(s)

                    avg_polarity_n = (sum(sentiments_n) / len(sentiments_n)) if sentiments_n else 0.0
                    auto_weight_n = compute_raw_weight(base_weight, avg_polarity_n, index_type="N")
                    rows_n.append({
                        "symbol": symbol,
                        "name": name,
                        "base_weight": base_weight,
                        "num_articles": len(sentiments_n),
                        "avg_polarity": avg_polarity_n,
                        "auto_weight": auto_weight_n
                    })
                    step += 1
                    progress.progress(int(step / total_steps * 100))
                    log_container.info(f"[{i+1}/{len(df_input)}] {symbol} — Flash-N: {len(sentiments_n)} articles, avg_pol={avg_polarity_n:.4f}")

                    success_count += 1

                except Exception as e:
                    # log error but continue
                    st.warning(f"Error processing row {i} ({row.get('symbol','')}) — continuing. Error: {e}")
                    step += 2  # skip both P & N for this row
                    progress.progress(min(100, int(step / total_steps * 100)))
                    continue

            # Build DataFrames and normalize
            df_p = pd.DataFrame(rows_p) if rows_p else pd.DataFrame(columns=["symbol","name","base_weight","num_articles","avg_polarity","auto_weight"])
            df_n = pd.DataFrame(rows_n) if rows_n else pd.DataFrame(columns=["symbol","name","base_weight","num_articles","avg_polarity","auto_weight"])

            if len(df_p) == 0 and len(df_n) == 0:
                st.error("No results were generated. Check your API keys, CSE configuration, and CSV contents.")
            else:
                if not df_p.empty:
                    df_p = normalize_weights(df_p, "auto_weight")
                    df_p["avg_polarity"] = df_p["avg_polarity"].round(4)
                    df_p["auto_weight"] = df_p["auto_weight"].round(6)
                    df_p["normalized_weight"] = df_p["normalized_weight"].round(6)

                if not df_n.empty:
                    df_n = normalize_weights(df_n, "auto_weight")
                    df_n["avg_polarity"] = df_n["avg_polarity"].round(4)
                    df_n["auto_weight"] = df_n["auto_weight"].round(6)
                    df_n["normalized_weight"] = df_n["normalized_weight"].round(6)

                st.success(f"✅ Indexes created for {success_count} symbols.")
                st.subheader("Flash-P (Positive) Index")
                if not df_p.empty:
                    st.dataframe(df_p.sort_values("normalized_weight", ascending=False).reset_index(drop=True))
                else:
                    st.info("Flash-P returned no rows.")

                st.subheader("Flash-N (Negative) Index")
                if not df_n.empty:
                    st.dataframe(df_n.sort_values("normalized_weight", ascending=False).reset_index(drop=True))
                else:
                    st.info("Flash-N returned no rows.")

                csv_p = df_p.to_csv(index=False).encode("utf-8") if not df_p.empty else None
                csv_n = df_n.to_csv(index=False).encode("utf-8") if not df_n.empty else None

                if csv_p:
                    st.download_button("📥 Download flash_p_index.csv", data=csv_p, file_name="flash_p_index.csv", mime="text/csv")
                if csv_n:
                    st.download_button("📥 Download flash_n_index.csv", data=csv_n, file_name="flash_n_index.csv", mime="text/csv")

                # Plots
                def plot_top10(df, title):
                    if df.empty:
                        st.info(f"No data to plot for {title}")
                        return
                    top = df.sort_values("normalized_weight", ascending=False).head(10)
                    plt.figure(figsize=(10, 4))
                    plt.bar(top["symbol"].astype(str), top["normalized_weight"])
                    plt.title(title)
                    plt.xlabel("Symbol")
                    plt.ylabel("Normalized Weight")
                    st.pyplot(plt)

                st.subheader("Visuals")
                col1, col2 = st.columns(2)
                with col1:
                    plot_top10(df_p, "Flash-P Top 10")
                with col2:
                    plot_top10(df_n, "Flash-N Top 10")

                st.markdown(
                    """
                    **Columns explanation**
                    - `base_weight`: weight supplied in your input CSV.
                    - `num_articles`: number of articles successfully scraped/analyzed for that symbol by the engine.
                    - `avg_polarity`: average TextBlob polarity from -1 (very negative) to +1 (very positive).
                    - `auto_weight`: raw auto-weight computed from base_weight and avg_polarity (before normalization).
                    - `normalized_weight`: index weight normalized to sum to 1 across the index.
                    """
                )
                st.balloons()
