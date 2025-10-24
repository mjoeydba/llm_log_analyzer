"""Simple Tkinter-based desktop UI for the log analyzer.

The desktop application exposes a subset of the Streamlit workflow so that
analysts can work without launching a web browser. It reuses the shared helper
functions defined in :mod:`core` for Elasticsearch access, summarisation and
Ollama prompting.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime, timedelta, timezone
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Optional

import pandas as pd
from dateutil import parser as dateparser

from core import (
    build_query,
    build_single_period_prompt,
    connect_es,
    fetch_dataframe,
    isoformat,
    ollama_generate,
    summarize_dataframe,
)


CST = timezone(timedelta(hours=-6), name="CST")


class DesktopApp:
    """Encapsulates the Tkinter widgets and interactions."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Elastic Data Explorer (Desktop)")

        self.es_client = None
        self.sample_df: pd.DataFrame = pd.DataFrame()
        self.current_df: pd.DataFrame = pd.DataFrame()
        self.available_fields: List[str] = []

        self._build_layout()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Connection frame
        conn_frame = ttk.LabelFrame(main, text="Elasticsearch Connection", padding=10)
        conn_frame.grid(row=0, column=0, sticky="nsew")

        self.host_var = tk.StringVar(value="http://localhost:9200")
        self.index_var = tk.StringVar(value="logs-*")
        self.time_field_var = tk.StringVar(value="@timestamp")
        self.query_var = tk.StringVar(value="*")

        ttk.Label(conn_frame, text="Host").grid(row=0, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.host_var, width=40).grid(
            row=0, column=1, sticky="ew"
        )

        ttk.Label(conn_frame, text="Index / pattern").grid(row=1, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.index_var, width=30).grid(
            row=1, column=1, sticky="ew"
        )

        ttk.Label(conn_frame, text="Time field").grid(row=2, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.time_field_var, width=30).grid(
            row=2, column=1, sticky="ew"
        )

        ttk.Label(conn_frame, text="Lucene query").grid(row=3, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.query_var, width=50).grid(
            row=3, column=1, sticky="ew"
        )

        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        self.ca_certs_var = tk.StringVar()
        self.auth_mode = tk.StringVar(value="none")

        ttk.Label(conn_frame, text="Auth mode").grid(row=4, column=0, sticky="w")
        auth_menu = ttk.Combobox(
            conn_frame,
            textvariable=self.auth_mode,
            values=("none", "basic", "api_key"),
            state="readonly",
        )
        auth_menu.grid(row=4, column=1, sticky="ew")

        ttk.Label(conn_frame, text="Username").grid(row=5, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.username_var).grid(
            row=5, column=1, sticky="ew"
        )
        ttk.Label(conn_frame, text="Password").grid(row=6, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.password_var, show="*").grid(
            row=6, column=1, sticky="ew"
        )
        ttk.Label(conn_frame, text="API key").grid(row=7, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.api_key_var).grid(
            row=7, column=1, sticky="ew"
        )

        ttk.Label(conn_frame, text="CA bundle path").grid(row=8, column=0, sticky="w")
        ttk.Entry(conn_frame, textvariable=self.ca_certs_var, width=40).grid(
            row=8, column=1, sticky="ew"
        )

        ttk.Button(conn_frame, text="Connect", command=self.connect).grid(
            row=9, column=0, columnspan=2, pady=(8, 0)
        )

        conn_frame.columnconfigure(1, weight=1)

        # Time range & fetch frame
        fetch_frame = ttk.LabelFrame(main, text="Time range & Data", padding=10)
        fetch_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        now = datetime.now(tz=CST)
        default_start = now - timedelta(days=1)
        self.start_var = tk.StringVar(value=default_start.isoformat())
        self.end_var = tk.StringVar(value=now.isoformat())
        self.max_docs_var = tk.StringVar(value="5000")
        self.sample_size_var = tk.StringVar(value="1000")

        ttk.Label(fetch_frame, text="Start (ISO8601)").grid(row=0, column=0, sticky="w")
        ttk.Entry(fetch_frame, textvariable=self.start_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(fetch_frame, text="End (ISO8601)").grid(row=1, column=0, sticky="w")
        ttk.Entry(fetch_frame, textvariable=self.end_var).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Button(
            fetch_frame,
            text="Choose…",
            command=self._open_time_popup,
        ).grid(row=0, column=2, rowspan=2, padx=(8, 0))
        ttk.Label(fetch_frame, text="Max docs").grid(row=2, column=0, sticky="w")
        ttk.Entry(fetch_frame, textvariable=self.max_docs_var).grid(
            row=2, column=1, sticky="ew"
        )
        ttk.Label(fetch_frame, text="Sample size").grid(row=3, column=0, sticky="w")
        ttk.Entry(fetch_frame, textvariable=self.sample_size_var).grid(
            row=3, column=1, sticky="ew"
        )

        ttk.Button(fetch_frame, text="Fetch sample fields", command=self.fetch_sample).grid(
            row=4, column=0, columnspan=2, pady=(6, 0)
        )
        ttk.Button(fetch_frame, text="Fetch full data", command=self.fetch_data).grid(
            row=5, column=0, columnspan=2, pady=(6, 0)
        )

        fetch_frame.columnconfigure(1, weight=1)

        # Column selection
        cols_frame = ttk.LabelFrame(main, text="Available fields", padding=10)
        cols_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(10, 0))

        self.columns_list = tk.Listbox(cols_frame, selectmode=tk.EXTENDED, height=15)
        self.columns_list.grid(row=0, column=0, sticky="nsew")

        cols_frame.rowconfigure(0, weight=1)
        cols_frame.columnconfigure(0, weight=1)

        select_btns = ttk.Frame(cols_frame)
        select_btns.grid(row=1, column=0, pady=(6, 0), sticky="ew")
        ttk.Button(select_btns, text="Select all", command=self.select_all_columns).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(select_btns, text="Clear", command=self.clear_selection).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )

        # LLM frame
        llm_frame = ttk.LabelFrame(main, text="Ollama", padding=10)
        llm_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

        self.ollama_host_var = tk.StringVar(value="http://localhost:11434")
        self.model_var = tk.StringVar(value="llama3.1")
        self.temperature_var = tk.DoubleVar(value=0.2)
        self.max_tokens_var = tk.StringVar(value="0")

        ttk.Label(llm_frame, text="Host").grid(row=0, column=0, sticky="w")
        ttk.Entry(llm_frame, textvariable=self.ollama_host_var, width=30).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(llm_frame, text="Model").grid(row=1, column=0, sticky="w")
        ttk.Entry(llm_frame, textvariable=self.model_var).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Label(llm_frame, text="Temperature").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            llm_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.temperature_var,
        ).grid(row=2, column=1, sticky="ew")
        ttk.Label(llm_frame, text="Max tokens (0 = default)").grid(row=3, column=0, sticky="w")
        ttk.Entry(llm_frame, textvariable=self.max_tokens_var).grid(
            row=3, column=1, sticky="ew"
        )

        ttk.Label(llm_frame, text="Supplemental context").grid(
            row=4, column=0, sticky="nw"
        )
        self.supplemental_text = ScrolledText(llm_frame, height=4)
        self.supplemental_text.grid(row=4, column=1, sticky="nsew")

        ttk.Button(llm_frame, text="Analyze with Ollama", command=self.run_analysis).grid(
            row=5, column=0, columnspan=2, pady=(8, 0)
        )

        llm_frame.columnconfigure(1, weight=1)
        llm_frame.rowconfigure(4, weight=1)

        # Output frame
        output_frame = ttk.LabelFrame(main, text="Output", padding=10)
        output_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

        self.output_text = ScrolledText(output_frame, height=16)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Not connected")
        status = ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN)
        status.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        main.rowconfigure(3, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _duration_to_timedelta(self, value: str, unit: str) -> timedelta:
        try:
            amount = float(value)
        except ValueError:
            raise ValueError("Enter a numeric duration") from None

        if amount < 0:
            raise ValueError("Duration must be non-negative")

        unit = unit.lower()
        if unit in {"minute", "minutes"}:
            return timedelta(minutes=amount)
        if unit in {"hour", "hours"}:
            return timedelta(hours=amount)
        if unit in {"day", "days"}:
            return timedelta(days=amount)
        if unit in {"week", "weeks"}:
            return timedelta(weeks=amount)

        raise ValueError(f"Unsupported unit: {unit}")

    def _open_time_popup(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Select time range")
        popup.transient(self.root)
        popup.grab_set()

        mode_var = tk.StringVar(value="absolute")

        container = ttk.Frame(popup, padding=10)
        container.grid(row=0, column=0, sticky="nsew")

        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        ttk.Label(container, text="Mode").grid(row=0, column=0, sticky="w")
        mode_frame = ttk.Frame(container)
        mode_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Absolute",
            value="absolute",
            variable=mode_var,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Radiobutton(
            mode_frame,
            text="Relative",
            value="relative",
            variable=mode_var,
        ).grid(row=0, column=1)

        abs_frame = ttk.LabelFrame(container, text="Absolute range (CST)", padding=10)
        abs_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        abs_start_var = tk.StringVar(value=self.start_var.get())
        abs_end_var = tk.StringVar(value=self.end_var.get())

        ttk.Label(abs_frame, text="Start").grid(row=0, column=0, sticky="w")
        ttk.Entry(abs_frame, textvariable=abs_start_var, width=35).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(abs_frame, text="End").grid(row=1, column=0, sticky="w")
        ttk.Entry(abs_frame, textvariable=abs_end_var, width=35).grid(
            row=1, column=1, sticky="ew"
        )

        abs_frame.columnconfigure(1, weight=1)

        rel_frame = ttk.LabelFrame(container, text="Relative range (from now)", padding=10)
        rel_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        rel_length_var = tk.StringVar(value="1")
        rel_length_unit_var = tk.StringVar(value="days")
        rel_offset_var = tk.StringVar(value="0")
        rel_offset_unit_var = tk.StringVar(value="minutes")

        ttk.Label(rel_frame, text="Duration").grid(row=0, column=0, sticky="w")
        ttk.Entry(rel_frame, textvariable=rel_length_var, width=8).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Combobox(
            rel_frame,
            textvariable=rel_length_unit_var,
            values=("minutes", "hours", "days", "weeks"),
            state="readonly",
            width=10,
        ).grid(row=0, column=2, padx=(6, 0))

        ttk.Label(rel_frame, text="End offset").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(rel_frame, textvariable=rel_offset_var, width=8).grid(
            row=1, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Combobox(
            rel_frame,
            textvariable=rel_offset_unit_var,
            values=("minutes", "hours", "days", "weeks"),
            state="readonly",
            width=10,
        ).grid(row=1, column=2, padx=(6, 0), pady=(6, 0))

        rel_frame.columnconfigure(3, weight=1)

        def _set_state(frame: ttk.LabelFrame, enabled: bool) -> None:
            for child in frame.winfo_children():
                try:
                    if isinstance(child, ttk.Combobox):
                        child.configure(state="readonly" if enabled else "disabled")
                    else:
                        child.configure(state="normal" if enabled else "disabled")
                except tk.TclError:
                    # Widgets like labels might not support state changes; ignore.
                    pass

        def update_mode() -> None:
            use_absolute = mode_var.get() == "absolute"
            _set_state(abs_frame, use_absolute)
            _set_state(rel_frame, not use_absolute)

        update_mode()
        mode_var.trace_add("write", lambda *_: update_mode())

        def apply_time_range() -> None:
            try:
                if mode_var.get() == "absolute":
                    start_text = abs_start_var.get().strip()
                    end_text = abs_end_var.get().strip()
                    if not start_text or not end_text:
                        raise ValueError("Both start and end are required")

                    start_dt = dateparser.parse(start_text)
                    end_dt = dateparser.parse(end_text)
                    if start_dt is None or end_dt is None:
                        raise ValueError("Unable to parse provided timestamps")

                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=CST)
                    else:
                        start_dt = start_dt.astimezone(CST)

                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=CST)
                    else:
                        end_dt = end_dt.astimezone(CST)

                else:
                    duration = self._duration_to_timedelta(
                        rel_length_var.get(), rel_length_unit_var.get()
                    )
                    end_offset = self._duration_to_timedelta(
                        rel_offset_var.get(), rel_offset_unit_var.get()
                    )

                    now_cst = datetime.now(tz=CST)
                    end_dt = now_cst - end_offset
                    start_dt = end_dt - duration

                if start_dt >= end_dt:
                    raise ValueError("Start must be before end")

            except ValueError as exc:
                messagebox.showerror("Invalid time range", str(exc), parent=popup)
                return

            self.start_var.set(start_dt.isoformat())
            self.end_var.set(end_dt.isoformat())
            popup.destroy()

        buttons = ttk.Frame(container)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=popup.destroy).grid(row=0, column=0)
        ttk.Button(buttons, text="Apply", command=apply_time_range).grid(
            row=0, column=1, padx=(8, 0)
        )

        popup.wait_window()

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        if not value.strip():
            return None
        dt = dateparser.parse(value)
        if not dt:
            raise ValueError(f"Could not parse datetime: {value}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CST)
        else:
            dt = dt.astimezone(CST)
        return dt

    def _selected_columns(self) -> List[str]:
        selection = [self.columns_list.get(i) for i in self.columns_list.curselection()]
        if selection:
            return selection
        return self.available_fields

    def _current_query(self):
        start_dt = self._parse_datetime(self.start_var.get())
        end_dt = self._parse_datetime(self.end_var.get())
        start_iso = isoformat(start_dt) if start_dt else None
        end_iso = isoformat(end_dt) if end_dt else None
        return build_query(
            self.query_var.get(), self.time_field_var.get() or None, start_iso, end_iso
        )

    def _require_es(self) -> bool:
        if not self.es_client:
            messagebox.showwarning("Elasticsearch", "Connect to Elasticsearch first.")
            return False
        return True

    def _update_output(self, text: str) -> None:
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def connect(self) -> None:
        host = self.host_var.get().strip()
        username = password = api_key = None
        mode = self.auth_mode.get()
        if mode == "basic":
            username = self.username_var.get().strip() or None
            password = self.password_var.get()
        elif mode == "api_key":
            api_key = self.api_key_var.get().strip() or None

        ca_bundle = self.ca_certs_var.get().strip() or None
        verify_certs = bool(ca_bundle)

        try:
            self.es_client = connect_es(
                host=host,
                username=username,
                password=password,
                api_key=api_key,
                verify_certs=verify_certs,
                ca_certs_path=ca_bundle,
            )
            self.status_var.set("Connected to Elasticsearch")
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Connection failed", str(exc))
            self.status_var.set("Connection failed")

    def fetch_sample(self) -> None:
        if not self._require_es():
            return
        try:
            query = self._current_query()
            sample_size = int(self.sample_size_var.get())
            df = fetch_dataframe(
                es=self.es_client,
                index=self.index_var.get().strip(),
                query=query,
                source_includes=None,
                max_docs=sample_size,
                sample_only=True,
                sample_size=sample_size,
            )
            self.sample_df = df
            self.available_fields = list(df.columns)
            self.columns_list.delete(0, tk.END)
            for col in self.available_fields:
                self.columns_list.insert(tk.END, col)
            if not self.available_fields:
                self._update_output("No fields discovered in the sample.")
            else:
                self._update_output(
                    f"Discovered {len(self.available_fields)} fields.\n"
                    + df.head(10).to_string(index=False)
                )
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Sample error", str(exc))

    def fetch_data(self) -> None:
        if not self._require_es():
            return
        try:
            query = self._current_query()
            max_docs = int(self.max_docs_var.get())
            selected = self._selected_columns()
            includes = list({*(selected or []), self.time_field_var.get()})
            df = fetch_dataframe(
                es=self.es_client,
                index=self.index_var.get().strip(),
                query=query,
                source_includes=[c for c in includes if c],
                max_docs=max_docs,
                sample_only=False,
            )
            self.current_df = df
            if df.empty:
                self._update_output("No documents matched the current settings.")
            else:
                preview = df.head(20).to_string(index=False)
                self._update_output(f"Fetched {len(df)} rows.\n\n{preview}")
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Fetch error", str(exc))

    def run_analysis(self) -> None:
        if self.current_df.empty:
            messagebox.showinfo("Analysis", "Fetch data before running the analysis.")
            return

        columns = self._selected_columns()
        time_field = self.time_field_var.get() or None
        try:
            summary = summarize_dataframe(self.current_df, columns, time_field)
            start_iso = self.start_var.get().strip() or "unknown"
            end_iso = self.end_var.get().strip() or "unknown"
            prompt = build_single_period_prompt(
                index=self.index_var.get().strip(),
                lucene_query=self.query_var.get().strip(),
                time_field=time_field or "(none)",
                start_iso=start_iso,
                end_iso=end_iso,
                chosen_cols=columns,
                summary=summary,
                supplemental=self.supplemental_text.get("1.0", tk.END).strip(),
            )
            max_tokens = int(self.max_tokens_var.get() or 0)
            response = ollama_generate(
                model=self.model_var.get().strip(),
                prompt=prompt,
                host=self.ollama_host_var.get().strip(),
                temperature=float(self.temperature_var.get()),
                max_tokens=None if max_tokens == 0 else max_tokens,
            )
            self._update_output(response)
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Analysis error", str(exc))

    # ------------------------------------------------------------------
    # Column selection helpers
    # ------------------------------------------------------------------
    def select_all_columns(self) -> None:
        self.columns_list.select_set(0, tk.END)

    def clear_selection(self) -> None:
        self.columns_list.selection_clear(0, tk.END)


def main() -> None:
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
