import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from config_store import load_saved_connection, save_connection_config
from core import (
    RequestCancelled,
    build_compare_prompt,
    build_query,
    build_single_period_prompt,
    connect_es,
    fetch_dataframe,
    isoformat,
    ollama_generate,
    prepare_compare_bundle,
    summarize_dataframe,
)

from logging_utils import (
    attach_elasticsearch_handler,
    detach_elasticsearch_handler,
    ensure_file_logger,
)


LOG_PATH = Path(os.environ.get("LLM_LOG_ANALYZER_LOG", Path.cwd() / "llm_log_analyzer.log"))
CONFIG_PATH = Path(
    os.environ.get(
        "LLM_LOG_ANALYZER_CONFIG",
        Path.home() / ".llm_log_analyzer_config.json",
    )
)
ensure_file_logger(LOG_PATH)
logger = logging.getLogger("llm_log_analyzer.app")


saved_config = load_saved_connection(CONFIG_PATH)
if saved_config:
    logger.info("Loaded Elasticsearch connection configuration from %s", CONFIG_PATH)

if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False

if "es_log_handler" not in st.session_state:
    st.session_state.es_log_handler = None

if "log_index_value" not in st.session_state:
    st.session_state.log_index_value = None


def request_cancel() -> None:
    st.session_state.cancel_requested = True
    logger.info("Cancellation requested by user")


def reset_cancel_flag() -> None:
    st.session_state.cancel_requested = False


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Elastic Data Explorer + Ollama Insights", layout="wide")

st.title("Elastic Data Explorer + Ollama Insights")

with st.sidebar:
    st.header("Elasticsearch Connection")
    host = st.text_input(
        "Host (e.g., https://es.example.com:9200 or http://localhost:9200)",
        value=saved_config.get("host", os.environ.get("ES_HOST", "http://localhost:9200")),
    )
    index = st.text_input(
        "Index or index pattern",
        value=saved_config.get("index", os.environ.get("ES_INDEX", "logs-*")),
    )
    time_field = st.text_input(
        "Time field",
        value=saved_config.get("time_field", os.environ.get("ES_TIME_FIELD", "@timestamp")),
    )

    st.markdown("**Authentication**")
    auth_options = ["None", "Basic (username/password)", "API key"]
    saved_auth = saved_config.get("auth_method")
    default_idx = auth_options.index(saved_auth) if saved_auth in auth_options else 1
    auth_method = st.selectbox("Auth method", auth_options, index=default_idx)
    username = password = api_key = None
    if auth_method == "Basic (username/password)":
        username = st.text_input(
            "Username",
            value=saved_config.get("username", os.environ.get("ES_USERNAME", "elastic")),
        )
        password = st.text_input(
            "Password",
            type="password",
            value=saved_config.get("password", os.environ.get("ES_PASSWORD", "")),
        )
    elif auth_method == "API key":
        api_key = st.text_input(
            "API key (id:secret or base64 value)",
            value=saved_config.get("api_key", os.environ.get("ES_API_KEY", "")),
        )

    verify_certs = st.checkbox(
        "Verify SSL certificates",
        value=bool(saved_config.get("verify_certs", False)),
    )
    ca_certs_path = st.text_input(
        "CA certs path (optional)",
        value=saved_config.get("ca_certs_path", ""),
    )

    log_index = st.text_input(
        "Log index for app logs (optional)",
        value=saved_config.get("log_index", os.environ.get("ES_LOG_INDEX", "")),
        help="If provided, application logs will also be indexed into this Elasticsearch index.",
    )

    if st.button("Save connection settings"):
        try:
            save_connection_config(
                {
                    "host": host,
                    "index": index,
                    "time_field": time_field,
                    "auth_method": auth_method,
                    "username": username or "",
                    "password": password or "",
                    "api_key": api_key or "",
                    "verify_certs": verify_certs,
                    "ca_certs_path": ca_certs_path,
                    "log_index": log_index,
                },
                CONFIG_PATH,
            )
            st.success(f"Connection settings saved to {CONFIG_PATH}")
        except Exception as exc:
            st.error(f"Failed to save settings: {exc}")

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
    st.button("Stop active request", on_click=request_cancel, help="Signal any in-flight Elasticsearch request to stop early.")
    if st.session_state.cancel_requested:
        st.info("Cancellation requested. New requests will reset this state when they start.")

if auth_method != "Basic (username/password)":
    username = None
    password = None
if auth_method != "API key":
    api_key = None

# Connection
conn_err = None
es_client = None
if host and index:
    try:
        logger.info(
            "Attempting Elasticsearch connection host=%s index=%s auth_method=%s",
            host,
            index,
            auth_method,
        )
        es_client = connect_es(
            host=host,
            username=username,
            password=password,
            api_key=api_key,
            verify_certs=verify_certs,
            ca_certs_path=ca_certs_path or None
        )
        logger.info("Elasticsearch connection established host=%s", host)
    except Exception as e:
        conn_err = str(e)
        logger.exception("Failed to connect to Elasticsearch host=%s", host)

if conn_err:
    st.error(f"Failed to connect to Elasticsearch: {conn_err}")
elif es_client is None:
    st.warning("Enter connection details to connect to Elasticsearch.")
else:
    st.success("Connected to Elasticsearch.")

root_logger = logging.getLogger("llm_log_analyzer")
current_handler = st.session_state.get("es_log_handler")
stored_index = st.session_state.get("log_index_value")

if es_client and log_index:
    if stored_index and stored_index != log_index and current_handler:
        detach_elasticsearch_handler(root_logger, current_handler)
        st.session_state.es_log_handler = None
        current_handler = None

    try:
        handler = attach_elasticsearch_handler(root_logger, es_client, log_index)
        st.session_state.es_log_handler = handler
        st.session_state.log_index_value = log_index
        if handler is not current_handler:
            logger.info(
                "Streaming application logs to Elasticsearch index %s", log_index
            )
    except Exception as exc:
        logger.exception("Failed to attach Elasticsearch logging handler")
        st.warning(f"Unable to stream app logs to Elasticsearch: {exc}")
else:
    if current_handler:
        detach_elasticsearch_handler(root_logger, current_handler)
        st.session_state.es_log_handler = None
        st.session_state.log_index_value = None

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
    time_mode = st.radio(
        "Time window mode",
        ["Absolute range", "Relative to issue time"],
        index=0,
        help="Choose between a traditional start/end window or a window relative to when an issue occurred."
    )

    start_dt = end_dt = issue_dt = None

    if time_mode == "Absolute range":
        col_abs_dates = st.columns(2)
        with col_abs_dates[0]:
            start_date = st.date_input(
                "Start date",
                value=default_start.date(),
                key="abs_start_date",
                help="Click to open a calendar picker for the range start."
            )
        with col_abs_dates[1]:
            end_date = st.date_input(
                "End date",
                value=now.date(),
                key="abs_end_date",
                help="Click to open a calendar picker for the range end."
            )

        # To capture precise times, add time inputs
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            start_time = st.time_input("Start time (UTC)", value=default_start.time(), key="abs_start_time")
        with col_t2:
            end_time = st.time_input("End time (UTC)", value=now.time(), key="abs_end_time")

        try:
            start_dt = datetime.combine(start_date, start_time, tzinfo=timezone.utc)
            end_dt = datetime.combine(end_date, end_time, tzinfo=timezone.utc)
        except Exception:
            start_dt = end_dt = None

    else:
        issue_date = st.date_input(
            "Issue date",
            value=now.date(),
            key="issue_date"
        )
        issue_time = st.time_input(
            "Issue time (UTC)",
            value=now.time(),
            key="issue_time"
        )
        relative_input_mode = st.radio(
            "Relative window input",
            ["Number fields", "Slider"],
            horizontal=True,
            key="relative_input_mode",
            help="Choose between numeric inputs or a slider to control the minutes before/after the issue time."
        )

        if relative_input_mode == "Slider":
            slider_min = -24 * 60
            slider_max = 24 * 60
            offset_range = st.slider(
                "Window offsets (minutes)",
                min_value=slider_min,
                max_value=slider_max,
                value=(-60, 60),
                step=5,
                key="issue_window_slider",
                help="Drag the handles to pick how many minutes before and after the issue time to search."
            )
            minutes_before = abs(min(offset_range[0], 0))
            minutes_after = max(offset_range[1], 0)
        else:
            minutes_before = st.number_input(
                "Minutes before issue",
                min_value=0,
                value=60,
                step=5,
                key="minutes_before_issue"
            )
            minutes_after = st.number_input(
                "Minutes after issue",
                min_value=0,
                value=60,
                step=5,
                key="minutes_after_issue"
            )

        try:
            issue_dt = datetime.combine(issue_date, issue_time, tzinfo=timezone.utc)
            start_dt = issue_dt - timedelta(minutes=int(minutes_before))
            end_dt = issue_dt + timedelta(minutes=int(minutes_after))
            st.caption(
                f"Searching from {start_dt.isoformat()} to {end_dt.isoformat()} (relative to issue time)."
            )
        except Exception:
            start_dt = end_dt = None

if start_dt and end_dt and start_dt >= end_dt:
    st.error("Start time must be before end time. Adjust your selections.")
    start_iso = end_iso = None
else:
    try:
        start_iso = isoformat(start_dt) if start_dt else None
        end_iso = isoformat(end_dt) if end_dt else None
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
    logger.info("Fetching field discovery sample index=%s sample_size=%s", index, sample_size)
    reset_cancel_flag()
    try:
        q = build_query(lucene_query, time_field, start_iso, end_iso)
        sample_df = fetch_dataframe(
            es=es_client,
            index=index,
            query=q,
            source_includes=None,  # get all for discovery
            max_docs=sample_size,
            sample_only=True,
            sample_size=sample_size,
            cancel_callback=lambda: st.session_state.get("cancel_requested", False),
        )
        st.session_state.sample_df = sample_df
        st.session_state.available_fields = list(sample_df.columns)
        if not st.session_state.available_fields:
            st.warning("No fields found in sample for the current query/time window.")
        else:
            st.success(f"Discovered {len(st.session_state.available_fields)} fields from the sample.")
    except RequestCancelled:
        logger.info("Field discovery sample cancelled by user")
        st.warning("Field discovery sample cancelled.")
    except Exception as e:
        logger.exception("Failed to fetch field discovery sample")
        st.error(f"Failed to fetch sample: {e}")
    finally:
        reset_cancel_flag()

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
    logger.info(
        "Fetching filtered data index=%s max_docs=%s chosen_cols=%s",
        index,
        max_docs,
        len(chosen_cols),
    )
    reset_cancel_flag()
    try:
        q = build_query(lucene_query, time_field, start_iso, end_iso)
        source_includes = list(set(chosen_cols + ([time_field] if time_field else [])))
        df = fetch_dataframe(
            es=es_client,
            index=index,
            query=q,
            source_includes=source_includes,
            max_docs=max_docs,
            sample_only=False,
            cancel_callback=lambda: st.session_state.get("cancel_requested", False),
        )
        st.session_state.current_df = df
        if df.empty:
            st.warning("No data returned for the current settings.")
        else:
            st.success(f"Fetched {len(df)} rows.")
    except RequestCancelled:
        logger.info("Filtered data fetch cancelled by user")
        st.warning("Data fetch cancelled.")
    except Exception as e:
        logger.exception("Failed to fetch filtered data")
        st.error(f"Failed to fetch data: {e}")
    finally:
        reset_cancel_flag()

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
            logger.info("Sending single-period summary to Ollama model=%s", model_name)
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
            logger.exception("LLM analysis failed")

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
        logger.info(
            "Fetching comparison datasets index=%s max_docs=%s includes=%s",
            index,
            max_docs,
            len(includes),
        )
        reset_cancel_flag()
        cancel_cb = lambda: st.session_state.get("cancel_requested", False)
        df_a = fetch_dataframe(
            es_client,
            index,
            q_a,
            source_includes=includes,
            max_docs=max_docs,
            cancel_callback=cancel_cb,
        )
        df_b = fetch_dataframe(
            es_client,
            index,
            q_b,
            source_includes=includes,
            max_docs=max_docs,
            cancel_callback=cancel_cb,
        )

        if df_a.empty and df_b.empty:
            st.warning("Both periods returned no data.")
        else:
            st.success(f"Fetched A: {len(df_a)} rows | B: {len(df_b)} rows")

            bundle = prepare_compare_bundle(
                df_a,
                df_b,
                chosen_cols,
                time_field if time_field in includes else None,
            )
            cmp_prompt = build_compare_prompt(
                index=index,
                lucene_query=lucene_query,
                time_field=time_field,
                a_start_iso=a_start_iso,
                a_end_iso=a_end_iso,
                b_start_iso=b_start_iso,
                b_end_iso=b_end_iso,
                period_a_label=period_a_label,
                period_b_label=period_b_label,
                chosen_cols=chosen_cols,
                compare_bundle=bundle,
                supplemental=supplemental_context,
            )
            logger.info("Sending comparison summary to Ollama model=%s", model_name)
            resp = ollama_generate(
                model=model_name,
                prompt=cmp_prompt,
                host=ollama_host,
                temperature=float(temperature),
                max_tokens=(None if max_llm_tokens == 0 else int(max_llm_tokens)),
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
                    mime="text/csv",
                )
            with c2:
                st.download_button(
                    "Download Period B (CSV)",
                    data=df_b.to_csv(index=False).encode("utf-8"),
                    file_name="period_B.csv",
                    mime="text/csv",
                )

    except RequestCancelled:
        logger.info("Comparison fetch cancelled by user")
        st.warning("Comparison cancelled.")
    except Exception as e:
        logger.exception("Failed during comparison workflow")
        st.error(f"Comparison failed: {e}")
    finally:
        reset_cancel_flag()

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Elasticsearch, and Ollama (local LLM). No data is sent to external LLM services unless you change the Ollama host.")
