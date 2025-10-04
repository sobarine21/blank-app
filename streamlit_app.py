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

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG - Streamlit secrets
# You MUST add the following keys into Streamlit secrets:
# GOOGLE_API_KEY_FLASH_P
# GOOGLE_SEARCH_ENGINE_ID_FLASH_P
# GOOGLE_API_KEY_FLASH_N
# GOOGLE_SEARCH_ENGINE_ID_FLASH_N
# ---------------------------

API_KEY_P = st.secrets.get("GOOGLE_API_KEY_FLASH_P")
CX_P = st.secrets.get("GOOGLE_SEARCH_ENGINE_ID_FLASH_P")
API_KEY_N = st.secrets.get("GOOGLE_API_KEY_FLASH_N")
CX_N = st.secrets.get("GOOGLE_SEARCH_ENGINE_ID_FLASH_N")

if not all([API_KEY_P, CX_P, API_KEY_N, CX_N]):
    st.error("Please add the four required secrets: GOOGLE_API_KEY_FLASH_P, GOOGLE_SEARCH_ENGINE_ID_FLASH_P, GOOGLE_API_KEY_FLASH_N, GOOGLE_SEARCH_ENGINE_ID_FLASH_N")
    st.stop()

# UI
st.set_page_config(page_title="Flash-P & Flash-N Index Creator", layout="wide")
st.title("⚡ Flash-P & Flash-N — Dynamic Sentiment Index Creator")
st.markdown("""
Upload a CSV with headers: `symbol, Name, Weights`.  
This tool will search news using two Google Custom Search setups (one for Flash-P and one for Flash-N),
classify sentiment automatically, assign auto-weights, normalize the index and produce two CSVs:
- `flash_p_index.csv` (positive index)
- `flash_n_index.csv` (negative index)
""")

# Helper: sanitize header names
def normalize_header(h):
    return re.sub(r'\s+', '', h.strip().lower())

# Upload CSV
uploaded = st.file_uploader("Upload CSV (symbol, Name, Weights)", type=["csv"])
num_results_per_symbol = st.number_input("Results per symbol to fetch (per engine)", min_value=1, max_value=10, value=5, step=1)
sleep_between_requests = st.number_input("Sleep between web requests (seconds)", min_value=0.0, max_value=5.0, value=0.8, step=0.1)

# Preprocess text: minimal cleaning
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fetch and parse article text from url (attempt)
def fetch_article_text(url, timeout=8):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FlashIndexBot/1.0; +https://example.com)"
    }
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Prefer article/body paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        text = preprocess_text(text)
        # ensure reasonably long content
        if len(text.split()) < 40:
            return ""
        return text
    except Exception:
        return ""

# Get search results from Google Custom Search
def google_search(api_key, cx, query, num=5):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        # The API allows up to 10 results per call; we use num param
        res = service.cse().list(q=query, cx=cx, num=num).execute()
        items = res.get("items", []) or []
        # Each item typically has 'link', 'title', 'snippet'
        results = []
        for it in items:
            results.append({
                "title": it.get("title"),
                "link": it.get("link"),
                "snippet": it.get("snippet", "")
            })
        return results
    except Exception as e:
        # Return empty list on failure (don't crash the whole run)
        return []

# Sentiment scoring: uses TextBlob polarity (-1 to 1)
def sentiment_score(text):
    try:
        tb = TextBlob(text)
        return tb.sentiment.polarity  # -1 .. 1
    except Exception:
        return 0.0

# Auto weight calculation (raw) from sentiment
# For Flash-P: positive polarity increases weight; negative reduces it.
# For Flash-N: negative polarity magnitude increases weight; positive reduces it.
def compute_raw_weight(base_weight, avg_polarity, index_type="P"):
    # clamp polarity to [-1,1]
    p = max(-1.0, min(1.0, avg_polarity))
    if index_type == "P":
        # boost by (1 + p) but keep floor so negative polarity reduces weight rather than invert
        raw = base_weight * (1.0 + max(0.0, p))
    else:  # "N"
        raw = base_weight * (1.0 + max(0.0, -p))  # use magnitude of negative polarity
    # Ensure non-zero positive
    return max(1e-8, raw)

# Normalize weights to sum to 1
def normalize_weights(df, weight_col="auto_weight"):
    total = df[weight_col].sum()
    if total <= 0:
        # fallback: equal weights
        df["normalized_weight"] = 1.0 / len(df)
    else:
        df["normalized_weight"] = df[weight_col] / total
    return df

# Main processing when user clicks
if uploaded is not None:
    try:
        df_input = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Ensure headers exist
    header_map = {normalize_header(h): h for h in df_input.columns}
    if not all(k in header_map for k in ["symbol", "name", "weights"]):
        st.error("CSV must contain headers: symbol, Name, Weights (case-insensitive).")
        st.stop()

    # Normalize columns into expected names
    df_input = df_input.rename(columns={
        header_map["symbol"]: "symbol",
        header_map["name"]: "name",
        header_map["weights"]: "base_weight"
    })

    # Ensure numeric base_weight
    df_input["base_weight"] = pd.to_numeric(df_input["base_weight"], errors="coerce").fillna(0.0)
    st.success(f"Loaded {len(df_input)} symbols.")

    # Provide a preview
    st.subheader("Input Preview")
    st.dataframe(df_input.head(50))

    if st.button("▶️ Build Flash-P & Flash-N Indexes"):
        progress = st.progress(0)
        total = len(df_input) * 2  # we'll run searches for both P and N
        idx_count = 0

        # Containers for aggregated index rows
        rows_p = []
        rows_n = []

        # We'll fetch using the P engine for Flash-P, N engine for Flash-N
        for i, row in df_input.iterrows():
            symbol = str(row["symbol"]).strip()
            name = str(row["name"]).strip()
            base_weight = float(row["base_weight"])

            # Prepare a search query: include symbol and name + "news"
            query = f"{name} {symbol} news"

            # ------------------ Flash-P (use API_KEY_P / CX_P) ------------------
            search_results_p = google_search(API_KEY_P, CX_P, query, num=int(num_results_per_symbol))
            sentiments_p = []
            for res in search_results_p:
                url = res.get("link")
                time.sleep(sleep_between_requests)
                text = fetch_article_text(url)
                if not text:
                    # fallback to snippet if page scraping failed but snippet exists
                    text = res.get("snippet", "")
                if text:
                    # Only compute sentiment for probable English text
                    try:
                        if detect(text[:200]) != "en":
                            # if non-english, still run TextBlob (it might be poor) but mark down
                            pass
                    except Exception:
                        pass
                    s = sentiment_score(text)
                    sentiments_p.append(s)

            avg_polarity = sum(sentiments_p) / len(sentiments_p) if sentiments_p else 0.0

            auto_weight_p = compute_raw_weight(base_weight, avg_polarity, index_type="P")
            rows_p.append({
                "symbol": symbol,
                "name": name,
                "base_weight": base_weight,
                "num_articles": len(sentiments_p),
                "avg_polarity": avg_polarity,
                "auto_weight": auto_weight_p
            })
            idx_count += 1
            progress.progress(min(1.0, idx_count / total))

            # ------------------ Flash-N (use API_KEY_N / CX_N) ------------------
            search_results_n = google_search(API_KEY_N, CX_N, query, num=int(num_results_per_symbol))
            sentiments_n = []
            for res in search_results_n:
                url = res.get("link")
                time.sleep(sleep_between_requests)
                text = fetch_article_text(url)
                if not text:
                    text = res.get("snippet", "")
                if text:
                    try:
                        if detect(text[:200]) != "en":
                            pass
                    except Exception:
                        pass
                    s = sentiment_score(text)
                    sentiments_n.append(s)

            avg_polarity_n = sum(sentiments_n) / len(sentiments_n) if sentiments_n else 0.0
            auto_weight_n = compute_raw_weight(base_weight, avg_polarity_n, index_type="N")
            rows_n.append({
                "symbol": symbol,
                "name": name,
                "base_weight": base_weight,
                "num_articles": len(sentiments_n),
                "avg_polarity": avg_polarity_n,
                "auto_weight": auto_weight_n
            })
            idx_count += 1
            progress.progress(min(1.0, idx_count / total))

        # Build dataframes
        df_p = pd.DataFrame(rows_p)
        df_n = pd.DataFrame(rows_n)

        # Normalize auto weights to sum to 1 within each index
        df_p = normalize_weights(df_p, weight_col="auto_weight")
        df_n = normalize_weights(df_n, weight_col="auto_weight")

        # Round numeric columns for readability
        df_p["avg_polarity"] = df_p["avg_polarity"].round(4)
        df_p["auto_weight"] = df_p["auto_weight"].round(6)
        df_p["normalized_weight"] = df_p["normalized_weight"].round(6)

        df_n["avg_polarity"] = df_n["avg_polarity"].round(4)
        df_n["auto_weight"] = df_n["auto_weight"].round(6)
        df_n["normalized_weight"] = df_n["normalized_weight"].round(6)

        st.success("✅ Indexes created.")

        # Show top rows
        st.subheader("Flash-P (Positive) Index")
        st.dataframe(df_p.sort_values("normalized_weight", ascending=False).reset_index(drop=True))

        st.subheader("Flash-N (Negative) Index")
        st.dataframe(df_n.sort_values("normalized_weight", ascending=False).reset_index(drop=True))

        # Download buttons for CSVs
        csv_p = df_p.to_csv(index=False).encode("utf-8")
        csv_n = df_n.to_csv(index=False).encode("utf-8")

        st.download_button("📥 Download flash_p_index.csv", data=csv_p, file_name="flash_p_index.csv", mime="text/csv")
        st.download_button("📥 Download flash_n_index.csv", data=csv_n, file_name="flash_n_index.csv", mime="text/csv")

        # Plots: bar charts of top 10 by normalized weight
        def plot_top10(df, title):
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

        # Provide a short explanation of columns
        st.markdown("""
        **Columns explanation**
        - `base_weight`: weight supplied in your input CSV.
        - `num_articles`: number of articles successfully scraped/analyzed for that symbol by the engine.
        - `avg_polarity`: average TextBlob polarity from -1 (very negative) to +1 (very positive).
        - `auto_weight`: raw auto-weight computed from base_weight and avg_polarity (before normalization).
        - `normalized_weight`: index weight normalized to sum to 1 across the index.
        """)

        st.balloons()
