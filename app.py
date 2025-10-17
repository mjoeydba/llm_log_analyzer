import os
import json
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
from dateutil import parser as dateparser

# Elasticsearch client (8.x) still imports from 'elasticsearch'
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan


# -------------------------------
# Ollama integration
# -------------------------------

def ollama_generate(model: str,
                    prompt: str,
                    host: str = "http://localhost:11434",
                    temperature: float = 0.2,
                    max_tokens: Optional[int] = None) -> str:
    """
    Generate text from a local Ollama server.
    Tries the 'ollama' package if available; falls back to raw HTTP.

    Returns the response text or raises an exception on failure.
    """
    try:
        import ollama  # type: ignore
        client = ollama.Client(host=host.rstrip("/"))
        # Stream for responsiveness; aggregate output before returning
        stream = client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": temperature,
                **({"num_predict": max_tokens} if max_tokens else {})
            }
        )
        out = []
        for part in stream:
            out.append(part.get("response", ""))
        return "".join(out)

    except Exception:
        # Fallback to HTTP
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


# -------------------------------
# Elasticsearch helpers
# -------------------------------

def connect_es(
    host: str,
    username: Optional[str],
    password: Optional[str],
    api_key: Optional[str],
    verify_certs: bool,
    ca_certs_path: Optional[str] = None,
    request_timeout: int = 60
) -> Elasticsearch:
    """
    Create and return an Elasticsearch client. Supports either basic auth or API key.
    """
    kwargs = {
        "hosts": [host],
        "verify_certs": verify_certs,
        "request_timeout": request_timeout,
    }
    if ca_certs_path:
        kwargs["ca_certs"] = ca_certs_path

    # Prefer API key if provided; else basic auth; else no auth.
    if api_key:
        kwargs["api_key"] = api_key
    elif username:
        # elasticsearch-py 8.x uses 'basic_auth'; 7.x used 'http_auth'. Try both.
        kwargs["basic_auth"] = (username, password or "")
    # else no auth

    try:
        es = Elasticsearch(**kwargs)
        _ = es.info()
        return es
    except TypeError:
        # Older client signature fallback
        if "basic_auth" in kwargs:
            http_auth = kwargs.pop("basic_auth")
            kwargs["http_auth"] = http_auth
        es = Elasticsearch(**kwargs)
        _ = es.info()
        return es


def build_query(lucene_query: str,
                time_field: Optional[str],
                start_iso: Optional[str],
                end_iso: Optional[str]) -> Dict:
    """
    Build a bool query that combines:
      - an optional Lucene query_string
      - an optional time range filter
    """
    must = []
    filters = []
    if lucene_query and lucene_query.strip():
        must.append({"query_string": {"query": lucene_query}})

    if time_field and start_iso and end_iso:
        # Elasticsearch accepts ISO8601 with Z or offset
        filters.append({
            "range": {
                time_field: {
                    "gte": start_iso,
                    "lte": end_iso,
                    "format": "strict_date_optional_time"
                }
            }
        })

    query = {"bool": {}}
    if must:
        query["bool"]["must"] = must
    if filters:
        query["bool"]["filter"] = filters
    if not must and not filters:
        query = {"match_all": {}}
    return query


def fetch_dataframe(
    es: Elasticsearch,
    index: str,
    query: Dict,
    source_includes: Optional[List[str]],
    max_docs: int = 5000,
    sample_only: bool = False,
    sample_size: int = 1000
) -> pd.DataFrame:
    """
    Fetch documents from Elasticsearch and return as a flattened pandas DataFrame.
    Uses a scan/scroll helper to handle large results, but caps at max_docs.
    """
    size = sample_size if sample_only else min(1000, max_docs)
    body = {"query": query}
    if source_includes:
        body["_source"] = source_includes

    docs = []
    # Use scan to iterate
    for hit in scan(es, index=index, query=body, size=size, scroll="2m"):
        src = hit.get("_source", {})
        docs.append(src)
        if not sample_only and len(docs) >= max_docs:
            break
        if sample_only and len(docs) >= sample_size:
            break

    if not docs:
        return pd.DataFrame()

    # Flatten nested JSON
    df = pd.json_normalize(docs, sep=".")
    return df


def isoformat(dt: datetime) -> str:
    """
    Ensure datetime is timezone-aware and return ISO 8601 string with 'Z' if UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def pick_resample_freq(start: datetime, end: datetime) -> str:
    """
    Choose a reasonable resample frequency (for counts) based on window length.
    <= 7 days: hourly; <= 60 days: daily; else weekly.
    """
    delta = end - start
    days = delta.total_seconds() / 86400.0
    if days <= 7:
        return "1H"
    elif days <= 60:
        return "1D"
    else:
        return "1W"


# -------------------------------
# Data summarization
# -------------------------------

def summarize_dataframe(df: pd.DataFrame,
                        chosen_cols: List[str],
                        time_field: Optional[str]) -> Dict:
    """
    Produce compact numeric/categorical summaries + small sample.
    """
    out: Dict[str, any] = {}

    sub = df[chosen_cols].copy() if chosen_cols else df.copy()
    numeric_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
    categorical_cols = [c for c in sub.columns if c not in numeric_cols]

    # Basic numeric stats
    numeric_summary = {}
    if numeric_cols:
        desc = sub[numeric_cols].describe(include="all", datetime_is_numeric=True).to_dict()
        for stat, per_col in desc.items():
            for col, val in per_col.items():
                numeric_summary.setdefault(col, {})[stat] = val

        # Quantiles
        quantiles = sub[numeric_cols].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        for col, qvals in quantiles.items():
            numeric_summary.setdefault(col, {})["quantiles"] = qvals

    # Top categories for non-numeric (limited)
    categorical_summary = {}
    for col in categorical_cols:
        # Avoid exploding high-cardinality: only show top 10
        vc = sub[col].astype(str).value_counts(dropna=False).head(10)
        categorical_summary[col] = vc.to_dict()

    # A small sample of rows (safe size)
    sample_rows = sub.head(10).to_dict(orient="records")

    out["numeric_summary"] = numeric_summary
    out["categorical_summary"] = categorical_summary
    out["sample_rows"] = sample_rows

    # If time field included, provide count-by-interval
    if time_field and time_field in df.columns:
        ts = df[[time_field]].copy()
        ts[time_field] = pd.to_datetime(ts[time_field], errors="coerce", utc=True)
        ts = ts.dropna(subset=[time_field]).set_index(time_field).sort_index()
        if not ts.empty:
            start = ts.index.min().to_pydatetime()
            end = ts.index.max().to_pydatetime()
            freq = pick_resample_freq(start, end)
            counts = ts.resample(freq).size().rename("count").reset_index()
            # Keep at most 200 points to avoid bloated prompts
            if len(counts) > 200:
                counts = counts.iloc[-200:]
            out["timeseries_counts"] = counts.to_dict(orient="records")

    return out


def compare_numeric_stats(df_a: pd.DataFrame,
                          df_b: pd.DataFrame,
                          numeric_cols: List[str]) -> Dict:
    """
    Compute per-column comparisons for numeric columns:
    count, mean, std, median, min, max + deltas and a simple effect size.
    """
    comp = {}
    for col in numeric_cols:
        a = df_a[col].dropna() if col in df_a.columns else pd.Series(dtype=float)
        b = df_b[col].dropna() if col in df_b.columns else pd.Series(dtype=float)
        a_stats = {
            "count": int(a.shape[0]),
            "mean": float(a.mean()) if a.shape[0] else None,
            "std": float(a.std(ddof=1)) if a.shape[0] > 1 else None,
            "median": float(a.median()) if a.shape[0] else None,
            "min": float(a.min()) if a.shape[0] else None,
            "max": float(a.max()) if a.shape[0] else None,
        }
        b_stats = {
            "count": int(b.shape[0]),
            "mean": float(b.mean()) if b.shape[0] else None,
            "std": float(b.std(ddof=1)) if b.shape[0] > 1 else None,
            "median": float(b.median()) if b.shape[0] else None,
            "min": float(b.min()) if b.shape[0] else None,
            "max": float(b.max()) if b.shape[0] else None,
        }

        # Percent change & simple Cohen's d (pooled std)
        pct_change = None
        effect_size = None
        if a_stats["mean"] is not None and b_stats["mean"] is not None:
            if a_stats["mean"] != 0:
                pct_change = (b_stats["mean"] - a_stats["mean"]) / abs(a_stats["mean"])
            # Cohen's d (if std not zero)
            s1, s2 = a_stats["std"], b_stats["std"]
            n1, n2 = a_stats["count"], b_stats["count"]
            if s1 is not None and s2 is not None and n1 > 1 and n2 > 1:
                pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                if pooled and pooled > 0:
                    effect_size = (b_stats["mean"] - a_stats["mean"]) / pooled

        comp[col] = {
            "period_a": a_stats,
            "period_b": b_stats,
            "delta_mean": None if (a_stats["mean"] is None or b_stats["mean"] is None) else (b_stats["mean"] - a_stats["mean"]),
            "pct_change_mean": pct_change,
            "effect_size": effect_size
        }
    return comp


def prepare_compare_bundle(df_a: pd.DataFrame,
                           df_b: pd.DataFrame,
                           chosen_cols: List[str],
                           time_field: Optional[str]) -> Dict:
    """
    Bundle summaries and comparisons for two periods.
    """
    all_cols = chosen_cols if chosen_cols else list(set(df_a.columns).union(set(df_b.columns)))
    # Identify numeric vs categorical over union
    numeric_cols = [c for c in all_cols if c in df_a.columns and pd.api.types.is_numeric_dtype(df_a[c]) or
                    c in df_b.columns and pd.api.types.is_numeric_dtype(df_b[c])]
    numeric_cols = list(dict.fromkeys(numeric_cols))  # dedup preserving order
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    summary_a = summarize_dataframe(df_a, chosen_cols, time_field)
    summary_b = summarize_dataframe(df_b, chosen_cols, time_field)
    cmp_numeric = compare_numeric_stats(df_a, df_b, numeric_cols)

    return {
        "numeric_comparison": cmp_numeric,
        "categorical_top_values_period_a": {col: summary_a["categorical_summary"].get(col, {}) for col in categorical_cols},
        "categorical_top_values_period_b": {col: summary_b["categorical_summary"].get(col, {}) for col in categorical_cols},
        "sample_rows_period_a": summary_a["sample_rows"],
        "sample_rows_period_b": summary_b["sample_rows"],
        "timeseries_counts_period_a": summary_a.get("timeseries_counts"),
        "timeseries_counts_period_b": summary_b.get("timeseries_counts"),
    }


# -------------------------------
# Prompt building
# -------------------------------

def build_single_period_prompt(index: str,
                               lucene_query: str,
                               time_field: str,
                               start_iso: str,
                               end_iso: str,
                               chosen_cols: List[str],
                               summary: Dict,
                               supplemental: str) -> str:
    return f"""
You are a data analyst. The user extracted data from Elasticsearch.

Context:
- Index: {index}
- Time field: {time_field}
- Time window: {start_iso} to {end_iso}
- Lucene query: {lucene_query or "(none)"}
- Selected columns: {json.dumps(chosen_cols)}

Task:
1) Provide concise insights about this dataset.
2) Highlight any notable anomalies or outliers.
3) Suggest possible causes or next steps to investigate.

Supplemental context from user (may provide domain knowledge or hypotheses):
{supplemental or "(none)"}

Data summaries (JSON):
- Numeric summary (per column): {json.dumps(summary.get("numeric_summary", {}), default=str)[:4000]}
- Categorical top values: {json.dumps(summary.get("categorical_summary", {}), default=str)[:4000]}
- Sample rows: {json.dumps(summary.get("sample_rows", []), default=str)[:2000]}
- Timeseries counts: {json.dumps(summary.get("timeseries_counts", []), default=str)[:2000]}

Please produce clear bullet points and, if helpful, short tables. Keep the answer focused and actionable.
""".strip()


def build_compare_prompt(index: str,
                         lucene_query: str,
                         time_field: str,
                         a_start_iso: str,
                         a_end_iso: str,
                         b_start_iso: str,
                         b_end_iso: str,
                         period_a_label: str,
                         period_b_label: str,
                         chosen_cols: List[str],
                         compare_bundle: Dict,
                         supplemental: str) -> str:
    return f"""
You are a data analyst. Compare TWO time periods from Elasticsearch data extracted with the SAME Lucene filter.

Context:
- Index: {index}
- Time field: {time_field}
- Lucene query: {lucene_query or "(none)"}

Periods:
- Period A: {a_start_iso} to {a_end_iso} (label: {period_a_label})
- Period B: {b_start_iso} to {b_end_iso} (label: {period_b_label})
- Selected columns: {json.dumps(chosen_cols)}

Task:
1) Compare the two periods and identify significant changes.
2) Point out anomalies in B vs A (or A vs B) — rate severity and likely impact.
3) Use labels (good/bad/unknown) as hints for which period represents a healthy baseline.
4) Suggest concrete investigative next steps and potential root causes.

Supplemental context from user:
{supplemental or "(none)"}

Comparative data (JSON):
- Numeric comparisons: {json.dumps(compare_bundle.get("numeric_comparison", {}), default=str)[:4000]}
- Categorical top values (A): {json.dumps(compare_bundle.get("categorical_top_values_period_a", {}), default=str)[:2000]}
- Categorical top values (B): {json.dumps(compare_bundle.get("categorical_top_values_period_b", {}), default=str)[:2000]}
- Timeseries counts (A): {json.dumps(compare_bundle.get("timeseries_counts_period_a", []), default=str)[:2000]}
- Timeseries counts (B): {json.dumps(compare_bundle.get("timeseries_counts_period_b", []), default=str)[:2000]}

Format:
- Use bullet points for findings.
- Include a small table of the top 3 most concerning anomalies with metric, direction, magnitude (% change or effect size), and suggested next step.
""".strip()


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Elastic Data Explorer + Ollama Insights", layout="wide")

st.title("Elastic Data Explorer + Ollama Insights")

with st.sidebar:
    st.header("Elasticsearch Connection")
    host = st.text_input("Host (e.g., https://es.example.com:9200 or http://localhost:9200)",
                         value=os.environ.get("ES_HOST", "http://localhost:9200"))
    index = st.text_input("Index or index pattern", value=os.environ.get("ES_INDEX", "logs-*"))
    time_field = st.text_input("Time field", value=os.environ.get("ES_TIME_FIELD", "@timestamp"))

    st.markdown("**Authentication**")
    auth_method = st.selectbox("Auth method", ["None", "Basic (username/password)", "API key"], index=1)
    username = password = api_key = None
    if auth_method == "Basic (username/password)":
        username = st.text_input("Username", value=os.environ.get("ES_USERNAME", "elastic"))
        password = st.text_input("Password", type="password", value=os.environ.get("ES_PASSWORD", ""))
    elif auth_method == "API key":
        api_key = st.text_input("API key (id:secret or base64 value)", value=os.environ.get("ES_API_KEY", ""))

    verify_certs = st.checkbox("Verify SSL certificates", value=False)
    ca_certs_path = st.text_input("CA certs path (optional)", value="")

    st.markdown("---")
    st.header("Ollama Settings")
    ollama_host = st.text_input("Ollama host", value=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    model_name = st.text_input("Model (e.g., llama3.1, mistral)", value=os.environ.get("OLLAMA_MODEL", "llama3.1"))
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_llm_tokens = st.number_input("Max tokens (optional, 0 = model default)", min_value=0, value=0, step=100)

    st.markdown("---")
    st.header("Fetch Settings")
    max_docs = st.number_input("Max docs to pull", min_value=100, max_value=500000, value=5000, step=100)
    sample_size = st.number_input("Sample size (for field discovery)", min_value=100, max_value=50000, value=1000, step=100)

# Connection
conn_err = None
es_client = None
if host and index:
    try:
        es_client = connect_es(
            host=host,
            username=username,
            password=password,
            api_key=api_key,
            verify_certs=verify_certs,
            ca_certs_path=ca_certs_path or None
        )
    except Exception as e:
        conn_err = str(e)

if conn_err:
    st.error(f"Failed to connect to Elasticsearch: {conn_err}")
elif es_client is None:
    st.warning("Enter connection details to connect to Elasticsearch.")
else:
    st.success("Connected to Elasticsearch.")

# -------------------------------
# Query + Time window inputs
# -------------------------------
st.header("1) Query / Time Window")

colq1, colq2 = st.columns([3, 2])
with colq1:
    lucene_query = st.text_area("Lucene query (optional)", value="", height=100,
                                help="Example: status:ERROR AND service:web*")

with colq2:
    now = datetime.now(timezone.utc)
    default_start = now - timedelta(days=1)
    window = st.date_input(
        "Base period dates (for single-period analysis)",
        value=(default_start.date(), now.date())
    )
    # To capture precise times, add time inputs
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_time = st.time_input("Start time (UTC)", value=default_start.time())
    with col_t2:
        end_time = st.time_input("End time (UTC)", value=now.time())

# Build ISO window for base period
try:
    start_dt = datetime.combine(window[0], start_time, tzinfo=timezone.utc)
    end_dt = datetime.combine(window[1], end_time, tzinfo=timezone.utc)
    start_iso = isoformat(start_dt)
    end_iso = isoformat(end_dt)
except Exception:
    start_iso = end_iso = None

# -------------------------------
# Pull sample + choose fields
# -------------------------------
st.header("2) Discover Fields (based on filtered data)")
colf1, colf2 = st.columns([1, 1])
with colf1:
    st.write("Click to pull a sample of data (using your query/time filter) to infer available fields.")
    refresh_fields = st.button("Refresh fields from sample")
with colf2:
    st.write("You can also export the sample to CSV after fetching.")
    export_sample = st.button("Export sample CSV")

if "available_fields" not in st.session_state:
    st.session_state.available_fields = []
if "sample_df" not in st.session_state:
    st.session_state.sample_df = pd.DataFrame()

if es_client and refresh_fields:
    try:
        q = build_query(lucene_query, time_field, start_iso, end_iso)
        sample_df = fetch_dataframe(
            es=es_client,
            index=index,
            query=q,
            source_includes=None,  # get all for discovery
            max_docs=sample_size,
            sample_only=True,
            sample_size=sample_size
        )
        st.session_state.sample_df = sample_df
        st.session_state.available_fields = list(sample_df.columns)
        if not st.session_state.available_fields:
            st.warning("No fields found in sample for the current query/time window.")
        else:
            st.success(f"Discovered {len(st.session_state.available_fields)} fields from the sample.")
    except Exception as e:
        st.error(f"Failed to fetch sample: {e}")

if export_sample:
    if st.session_state.sample_df is not None and not st.session_state.sample_df.empty:
        st.download_button(
            "Download sample CSV",
            data=st.session_state.sample_df.to_csv(index=False).encode("utf-8"),
            file_name="elastic_sample.csv",
            mime="text/csv"
        )
    else:
        st.warning("No sample data to export. Click 'Refresh fields from sample' first.")

chosen_cols = st.multiselect(
    "Choose columns to include (affects analysis and CSV export)",
    options=st.session_state.available_fields,
    default=[c for c in st.session_state.available_fields if c not in (time_field,)]
)

# -------------------------------
# Single-period analysis
# -------------------------------
st.header("3) Single-Period: Fetch Data + Analyze with Ollama")

col3a, col3b = st.columns([1, 1])
with col3a:
    fetch_data_btn = st.button("Fetch filtered data for current window")
with col3b:
    supplemental_context = st.text_area("Supplemental context for the LLM (optional)", height=120)

if "current_df" not in st.session_state:
    st.session_state.current_df = pd.DataFrame()

if es_client and fetch_data_btn:
    try:
        q = build_query(lucene_query, time_field, start_iso, end_iso)
        source_includes = list(set(chosen_cols + ([time_field] if time_field else [])))
        df = fetch_dataframe(
            es=es_client,
            index=index,
            query=q,
            source_includes=source_includes,
            max_docs=max_docs,
            sample_only=False
        )
        st.session_state.current_df = df
        if df.empty:
            st.warning("No data returned for the current settings.")
        else:
            st.success(f"Fetched {len(df)} rows.")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")

if not st.session_state.current_df.empty:
    st.dataframe(st.session_state.current_df.head(50))
    st.download_button(
        "Download filtered data (CSV)",
        data=st.session_state.current_df.to_csv(index=False).encode("utf-8"),
        file_name="elastic_filtered.csv",
        mime="text/csv"
    )

    analyze_now = st.button("Analyze this data with Ollama")
    if analyze_now:
        try:
            summary = summarize_dataframe(
                st.session_state.current_df,
                chosen_cols=chosen_cols,
                time_field=time_field if time_field in st.session_state.current_df.columns else None
            )
            prompt = build_single_period_prompt(
                index=index,
                lucene_query=lucene_query,
                time_field=time_field,
                start_iso=start_iso or "",
                end_iso=end_iso or "",
                chosen_cols=chosen_cols,
                summary=summary,
                supplemental=supplemental_context
            )
            resp = ollama_generate(
                model=model_name,
                prompt=prompt,
                host=ollama_host,
                temperature=float(temperature),
                max_tokens=(None if max_llm_tokens == 0 else int(max_llm_tokens))
            )
            st.subheader("LLM Insights")
            st.write(resp)
        except Exception as e:
            st.error(f"LLM analysis failed: {e}")

# -------------------------------
# Two-period comparison
# -------------------------------
st.header("4) Two-Period Comparison + Anomaly Hunt")

colp_a, colp_b = st.columns(2)
with colp_a:
    st.subheader("Period A")
    a_dates = st.date_input("Period A dates", value=( (now - timedelta(days=7)).date(), (now - timedelta(days=6)).date() ), key="a_dates")
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        a_start_time = st.time_input("A start time (UTC)", value=(now - timedelta(days=7)).time(), key="a_stime")
    with a_col2:
        a_end_time = st.time_input("A end time (UTC)", value=(now - timedelta(days=6)).time(), key="a_etime")
    period_a_label = st.selectbox("Label Period A", ["good", "bad", "unknown"], index=0)

with colp_b:
    st.subheader("Period B")
    b_dates = st.date_input("Period B dates", value=( (now - timedelta(days=1)).date(), now.date() ), key="b_dates")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        b_start_time = st.time_input("B start time (UTC)", value=(now - timedelta(days=1)).time(), key="b_stime")
    with b_col2:
        b_end_time = st.time_input("B end time (UTC)", value=now.time(), key="b_etime")
    period_b_label = st.selectbox("Label Period B", ["good", "bad", "unknown"], index=2)

compare_btn = st.button("Compare Periods with Ollama")

if compare_btn:
    try:
        a_start_dt = datetime.combine(a_dates[0], a_start_time, tzinfo=timezone.utc)
        a_end_dt = datetime.combine(a_dates[1], a_end_time, tzinfo=timezone.utc)
        b_start_dt = datetime.combine(b_dates[0], b_start_time, tzinfo=timezone.utc)
        b_end_dt = datetime.combine(b_dates[1], b_end_time, tzinfo=timezone.utc)

        a_start_iso, a_end_iso = isoformat(a_start_dt), isoformat(a_end_dt)
        b_start_iso, b_end_iso = isoformat(b_start_dt), isoformat(b_end_dt)

        # Build queries
        q_a = build_query(lucene_query, time_field, a_start_iso, a_end_iso)
        q_b = build_query(lucene_query, time_field, b_start_iso, b_end_iso)

        includes = list(set(chosen_cols + ([time_field] if time_field else [])))
        df_a = fetch_dataframe(es_client, index, q_a, source_includes=includes, max_docs=max_docs)
        df_b = fetch_dataframe(es_client, index, q_b, source_includes=includes, max_docs=max_docs)

        if df_a.empty and df_b.empty:
            st.warning("Both periods returned no data.")
        else:
            st.success(f"Fetched A: {len(df_a)} rows | B: {len(df_b)} rows")

            bundle = prepare_compare_bundle(df_a, df_b, chosen_cols, time_field if time_field in includes else None)
            cmp_prompt = build_compare_prompt(
                index=index,
                lucene_query=lucene_query,
                time_field=time_field,
                a_start_iso=a_start_iso, a_end_iso=a_end_iso,
                b_start_iso=b_start_iso, b_end_iso=b_end_iso,
                period_a_label=period_a_label,
                period_b_label=period_b_label,
                chosen_cols=chosen_cols,
                compare_bundle=bundle,
                supplemental=supplemental_context
            )
            resp = ollama_generate(
                model=model_name,
                prompt=cmp_prompt,
                host=ollama_host,
                temperature=float(temperature),
                max_tokens=(None if max_llm_tokens == 0 else int(max_llm_tokens))
            )
            st.subheader("LLM Comparison & Anomalies")
            st.write(resp)

            # Optional: allow downloads of the period data
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download Period A (CSV)",
                    data=df_a.to_csv(index=False).encode("utf-8"),
                    file_name="period_A.csv",
                    mime="text/csv"
                )
            with c2:
                st.download_button(
                    "Download Period B (CSV)",
                    data=df_b.to_csv(index=False).encode("utf-8"),
                    file_name="period_B.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Comparison failed: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Elasticsearch, and Ollama (local LLM). No data is sent to external LLM services unless you change the Ollama host.")
