import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan


def ollama_generate(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate text from a local Ollama server.

    Tries the ``ollama`` Python package first and falls back to a raw HTTP call
    when the package is unavailable. The function returns the generated text and
    raises on network or server errors.
    """

    try:
        import ollama  # type: ignore

        client = ollama.Client(host=host.rstrip("/"))
        stream = client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": temperature,
                **({"num_predict": max_tokens} if max_tokens else {}),
            },
        )
        out: List[str] = []
        for part in stream:
            out.append(part.get("response", ""))
        return "".join(out)

    except Exception:
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


def connect_es(
    host: str,
    username: Optional[str],
    password: Optional[str],
    api_key: Optional[str],
    verify_certs: bool,
    ca_certs_path: Optional[str] = None,
    request_timeout: int = 60,
) -> Elasticsearch:
    """Create and return an Elasticsearch client."""

    kwargs: Dict[str, object] = {
        "hosts": [host],
        "verify_certs": verify_certs,
        "request_timeout": request_timeout,
    }
    if ca_certs_path:
        kwargs["ca_certs"] = ca_certs_path

    if api_key:
        kwargs["api_key"] = api_key
    elif username:
        kwargs["basic_auth"] = (username, password or "")

    try:
        es = Elasticsearch(**kwargs)
        _ = es.info()
        return es
    except TypeError:
        if "basic_auth" in kwargs:
            http_auth = kwargs.pop("basic_auth")
            kwargs["http_auth"] = http_auth
        es = Elasticsearch(**kwargs)
        _ = es.info()
        return es


def build_query(
    lucene_query: str,
    time_field: Optional[str],
    start_iso: Optional[str],
    end_iso: Optional[str],
) -> Dict:
    """Build an Elasticsearch bool query combining the lucene filter and range."""

    must: List[Dict] = []
    filters: List[Dict] = []
    if lucene_query and lucene_query.strip():
        must.append({"query_string": {"query": lucene_query}})

    if time_field and start_iso and end_iso:
        filters.append(
            {
                "range": {
                    time_field: {
                        "gte": start_iso,
                        "lte": end_iso,
                        "format": "strict_date_optional_time",
                    }
                }
            }
        )

    query: Dict[str, Dict] = {"bool": {}}
    if must:
        query["bool"]["must"] = must
    if filters:
        query["bool"]["filter"] = filters
    if not must and not filters:
        return {"match_all": {}}
    return query


def fetch_dataframe(
    es: Elasticsearch,
    index: str,
    query: Dict,
    source_includes: Optional[List[str]],
    max_docs: int = 5000,
    sample_only: bool = False,
    sample_size: int = 1000,
) -> pd.DataFrame:
    """Fetch documents from Elasticsearch and return as a flattened DataFrame."""

    size = sample_size if sample_only else min(1000, max_docs)
    body = {"query": query}
    if source_includes:
        body["_source"] = source_includes

    docs: List[Dict] = []
    for hit in scan(es, index=index, query=body, size=size, scroll="2m"):
        src = hit.get("_source", {})
        docs.append(src)
        if not sample_only and len(docs) >= max_docs:
            break
        if sample_only and len(docs) >= sample_size:
            break

    if not docs:
        return pd.DataFrame()

    return pd.json_normalize(docs, sep=".")


def isoformat(dt: datetime) -> str:
    """Return an ISO8601 string for the provided datetime in UTC."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def pick_resample_freq(start: datetime, end: datetime) -> str:
    """Choose a resample frequency based on the window length."""

    delta = end - start
    days = delta.total_seconds() / 86400.0
    if days <= 7:
        return "1H"
    if days <= 60:
        return "1D"
    return "1W"


def summarize_dataframe(
    df: pd.DataFrame,
    chosen_cols: List[str],
    time_field: Optional[str],
) -> Dict:
    """Return numeric/categorical summaries and a sample slice."""

    out: Dict[str, object] = {}
    sub = df[chosen_cols].copy() if chosen_cols else df.copy()
    numeric_cols = [
        c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])
    ]
    categorical_cols = [c for c in sub.columns if c not in numeric_cols]

    numeric_summary: Dict[str, Dict[str, object]] = {}
    if numeric_cols:
        desc = sub[numeric_cols].describe(include="all", datetime_is_numeric=True).to_dict()
        for stat, per_col in desc.items():
            for col, val in per_col.items():
                numeric_summary.setdefault(col, {})[stat] = val

        quantiles = (
            sub[numeric_cols].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        )
        for col, qvals in quantiles.items():
            numeric_summary.setdefault(col, {})["quantiles"] = qvals

    categorical_summary: Dict[str, Dict[str, object]] = {}
    for col in categorical_cols:
        vc = sub[col].astype(str).value_counts(dropna=False).head(10)
        categorical_summary[col] = vc.to_dict()

    sample_rows = sub.head(10).to_dict(orient="records")

    out["numeric_summary"] = numeric_summary
    out["categorical_summary"] = categorical_summary
    out["sample_rows"] = sample_rows

    if time_field and time_field in df.columns:
        ts = df[[time_field]].copy()
        ts[time_field] = pd.to_datetime(ts[time_field], errors="coerce", utc=True)
        ts = ts.dropna(subset=[time_field]).set_index(time_field).sort_index()
        if not ts.empty:
            start = ts.index.min().to_pydatetime()
            end = ts.index.max().to_pydatetime()
            freq = pick_resample_freq(start, end)
            counts = ts.resample(freq).size().rename("count").reset_index()
            if len(counts) > 200:
                counts = counts.iloc[-200:]
            out["timeseries_counts"] = counts.to_dict(orient="records")

    return out


def compare_numeric_stats(
    df_a: pd.DataFrame, df_b: pd.DataFrame, numeric_cols: List[str]
) -> Dict:
    """Compute per-column comparisons for numeric metrics."""

    comp: Dict[str, Dict[str, object]] = {}
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

        pct_change = None
        effect_size = None
        if a_stats["mean"] is not None and b_stats["mean"] is not None:
            if a_stats["mean"] != 0:
                pct_change = (b_stats["mean"] - a_stats["mean"]) / abs(a_stats["mean"])
            s1, s2 = a_stats["std"], b_stats["std"]
            n1, n2 = a_stats["count"], b_stats["count"]
            if (
                s1 is not None
                and s2 is not None
                and n1 > 1
                and n2 > 1
            ):
                pooled = math.sqrt(
                    ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
                )
                if pooled and pooled > 0:
                    effect_size = (b_stats["mean"] - a_stats["mean"]) / pooled

        comp[col] = {
            "period_a": a_stats,
            "period_b": b_stats,
            "delta_mean": None
            if (
                a_stats["mean"] is None or b_stats["mean"] is None
            )
            else (b_stats["mean"] - a_stats["mean"]),
            "pct_change_mean": pct_change,
            "effect_size": effect_size,
        }
    return comp


def prepare_compare_bundle(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    chosen_cols: List[str],
    time_field: Optional[str],
) -> Dict:
    """Bundle summary information for two dataframes."""

    all_cols = (
        chosen_cols
        if chosen_cols
        else list(set(df_a.columns).union(set(df_b.columns)))
    )
    numeric_cols = [
        c
        for c in all_cols
        if (
            c in df_a.columns
            and pd.api.types.is_numeric_dtype(df_a[c])
        )
        or (
            c in df_b.columns
            and pd.api.types.is_numeric_dtype(df_b[c])
        )
    ]
    numeric_cols = list(dict.fromkeys(numeric_cols))
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    summary_a = summarize_dataframe(df_a, chosen_cols, time_field)
    summary_b = summarize_dataframe(df_b, chosen_cols, time_field)
    cmp_numeric = compare_numeric_stats(df_a, df_b, numeric_cols)

    return {
        "numeric_comparison": cmp_numeric,
        "categorical_top_values_period_a": {
            col: summary_a["categorical_summary"].get(col, {})
            for col in categorical_cols
        },
        "categorical_top_values_period_b": {
            col: summary_b["categorical_summary"].get(col, {})
            for col in categorical_cols
        },
        "sample_rows_period_a": summary_a["sample_rows"],
        "sample_rows_period_b": summary_b["sample_rows"],
        "timeseries_counts_period_a": summary_a.get("timeseries_counts"),
        "timeseries_counts_period_b": summary_b.get("timeseries_counts"),
    }


def build_single_period_prompt(
    index: str,
    lucene_query: str,
    time_field: str,
    start_iso: str,
    end_iso: str,
    chosen_cols: List[str],
    summary: Dict,
    supplemental: str,
) -> str:
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


def build_compare_prompt(
    index: str,
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
    supplemental: str,
) -> str:
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
