from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .constants import STRATEGY_LABELS


def _strategy_label_to_key_map() -> Dict[str, str]:
    return {STRATEGY_LABELS.get(k, k): k for k in STRATEGY_LABELS}


def _parse_index_spec(spec: str) -> List[int]:
    values = set()
    raw = (spec or "").strip()
    if not raw:
        return []

    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            left = left.strip()
            right = right.strip()
            if left.lstrip("-").isdigit() and right.lstrip("-").isdigit():
                start = int(left)
                end = int(right)
                step = 1 if end >= start else -1
                for idx in range(start, end + step, step):
                    values.add(idx)
            continue
        if token.lstrip("-").isdigit():
            values.add(int(token))

    return sorted(values)


def _make_unique_column_names(columns: List[object]) -> List[str]:
    seen: Dict[str, int] = {}
    unique: List[str] = []

    for raw_col in columns:
        base = str(raw_col)
        count = seen.get(base, 0)
        if count == 0:
            unique_name = base
        else:
            unique_name = f"{base}__{count + 1}"
        seen[base] = count + 1
        unique.append(unique_name)

    return unique


def _next_available_name(existing: List[str], base_name: str) -> str:
    if base_name not in existing:
        return base_name

    i = 2
    while f"{base_name}__{i}" in existing:
        i += 1
    return f"{base_name}__{i}"


def _collect_session_sources() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    sources: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []

    overview_df = st.session_state.get("overview_result")
    if isinstance(overview_df, pd.DataFrame) and not overview_df.empty:
        sources["Session - Overview"] = overview_df.copy()

    ensemble_df = st.session_state.get("ensemble_result")
    if isinstance(ensemble_df, pd.DataFrame) and not ensemble_df.empty:
        sources["Session - Ensemble"] = ensemble_df.copy()

    federated_bundle = st.session_state.get("federated_all_strategies_result")
    if isinstance(federated_bundle, dict):
        strategy_results = federated_bundle.get("strategy_results", {})
        round_frames: List[pd.DataFrame] = []
        final_frames: List[pd.DataFrame] = []
        label_map = _strategy_label_to_key_map()

        for strategy_label, result in strategy_results.items():
            strategy_key = label_map.get(strategy_label, strategy_label)
            if not isinstance(result, dict):
                continue

            round_df = result.get("user_round_metrics_df")
            if isinstance(round_df, pd.DataFrame) and not round_df.empty:
                tmp = round_df.copy()
                tmp["strategy"] = strategy_key
                round_frames.append(tmp)

            final_df = result.get("results_df")
            if isinstance(final_df, pd.DataFrame) and not final_df.empty:
                tmp = final_df.copy()
                tmp["strategy"] = strategy_key
                final_frames.append(tmp)

        if round_frames:
            sources["Session - Federated Round Metrics"] = pd.concat(round_frames, ignore_index=True)
        if final_frames:
            sources["Session - Federated Final Metrics"] = pd.concat(final_frames, ignore_index=True)
        if not round_frames and not final_frames:
            warnings.append("Federated results exist, but no plottable DataFrames were found.")

    return sources, warnings


def _draw_custom_plot(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
    color_col: Optional[str],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))

    if color_col and color_col in df.columns:
        labels = [v for v in df[color_col].dropna().astype(str).unique().tolist()]
        labels = sorted(labels)
        if not labels:
            labels = ["(all)"]

        for label in labels:
            if label == "(all)":
                subset = df.copy()
            else:
                subset = df[df[color_col].astype(str) == label].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(x_col, kind="stable")
            x_vals = subset[x_col]
            y_vals = subset[y_col].astype(float)
            if chart_type == "line":
                ax.plot(x_vals, y_vals, marker="o", linewidth=1.6, markersize=4.5, alpha=0.9, label=label)
            elif chart_type == "scatter":
                ax.scatter(x_vals, y_vals, alpha=0.85, s=30, label=label)
            else:
                ax.bar(x_vals, y_vals, alpha=0.8, label=label)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8, ncols=min(3, len(labels)), title=color_col)
    else:
        plot_df = df.sort_values(x_col, kind="stable")
        if chart_type == "line":
            ax.plot(
                plot_df[x_col],
                plot_df[y_col].astype(float),
                marker="o",
                linewidth=1.8,
                markersize=5,
                alpha=0.9,
                color="#2a9d8f",
            )
        elif chart_type == "scatter":
            ax.scatter(
                plot_df[x_col],
                plot_df[y_col].astype(float),
                alpha=0.85,
                s=34,
                color="#1f77b4",
            )
        else:
            ax.bar(
                plot_df[x_col],
                plot_df[y_col].astype(float),
                alpha=0.85,
                color="#f4a261",
            )

    ax.set_title(f"Custom {chart_type.title()} Plot")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3, linestyle="--")

    x_unique = df[x_col].nunique(dropna=True)
    if x_unique <= 30:
        ax.tick_params(axis="x", rotation=15)
    else:
        ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_custom_plotting_tab() -> None:
    st.subheader("Custom Plotting")
    st.caption(
        "Load experiment data from this session or upload CSV files, then build custom plots and"
        " hide problematic points non-destructively."
    )

    session_sources, session_warnings = _collect_session_sources()

    for msg in session_warnings:
        st.warning(msg)

    source_choices = ["Upload CSV"]
    source_choices.extend(session_sources.keys())

    source_choice = st.selectbox(
        "Data source",
        options=source_choices,
        index=0,
        help="Use CSV files from experiment exports or currently available session outputs.",
    )

    selected_df: Optional[pd.DataFrame] = None

    if source_choice == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload experiment CSV",
            type=["csv"],
            key="custom_plot_csv_uploader",
        )
        if uploaded is not None:
            try:
                selected_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")
                return
    else:
        selected_df = session_sources.get(source_choice)

    if selected_df is None or selected_df.empty:
        st.info("No data loaded yet. Upload a CSV or run an experiment first.")
        return

    work_df = selected_df.copy().reset_index(drop=True)
    work_df.columns = _make_unique_column_names(list(work_df.columns))

    row_id_col = _next_available_name(list(work_df.columns), "_row_id")
    work_df.insert(0, row_id_col, np.arange(len(work_df), dtype=int))

    st.markdown("### Configure Plot")
    numeric_cols = [c for c in work_df.columns if pd.api.types.is_numeric_dtype(work_df[c]) and c != row_id_col]
    if not numeric_cols:
        st.warning("The selected dataset has no numeric columns available for Y-axis plotting.")
        return

    default_x = "round" if "round" in work_df.columns else work_df.columns[0]
    default_y = "RMSE" if "RMSE" in numeric_cols else numeric_cols[0]

    control_c1, control_c2, control_c3 = st.columns(3)
    with control_c1:
        x_col = st.selectbox("X column", options=work_df.columns.tolist(), index=work_df.columns.tolist().index(default_x))
    with control_c2:
        y_col = st.selectbox("Y column", options=numeric_cols, index=numeric_cols.index(default_y))
    with control_c3:
        chart_type = st.selectbox("Chart type", options=["line", "scatter", "bar"], index=0)

    category_cols = [
        c
        for c in work_df.columns
        if c not in [row_id_col, x_col, y_col]
        and (
            pd.api.types.is_object_dtype(work_df[c])
            or pd.api.types.is_categorical_dtype(work_df[c])
            or pd.api.types.is_bool_dtype(work_df[c])
        )
    ]

    color_options = ["(none)"] + category_cols
    color_col_raw = st.selectbox("Color by", options=color_options, index=0)
    color_col = None if color_col_raw == "(none)" else color_col_raw

    st.markdown("### Issue Correction (Non-Destructive)")
    st.caption("Filters below only affect the corrected view and plot. Original source data remains unchanged.")

    corrected_df = work_df.copy()

    if "strategy" in corrected_df.columns:
        strategy_options = sorted(corrected_df["strategy"].dropna().astype(str).unique().tolist())
        strategy_default = strategy_options
        selected_strategies = st.multiselect(
            "Strategy filter",
            options=strategy_options,
            default=strategy_default,
            help="Primary filter for federated and strategy-based comparisons.",
        )
        if selected_strategies:
            corrected_df = corrected_df[corrected_df["strategy"].astype(str).isin(selected_strategies)].copy()
        else:
            corrected_df = corrected_df.iloc[0:0].copy()

    hide_nan = st.checkbox("Hide rows with missing X or Y", value=True)
    if hide_nan:
        corrected_df = corrected_df.dropna(subset=[x_col, y_col])

    index_spec = st.text_input(
        "Hide specific row ids (e.g. 3, 10-15)",
        value="",
        help=f"Use {row_id_col} values to remove known bad points from the corrected view.",
    )
    hidden_ids = _parse_index_spec(index_spec)
    if hidden_ids:
        corrected_df = corrected_df[~corrected_df[row_id_col].isin(hidden_ids)].copy()

    finite_y = pd.to_numeric(corrected_df[y_col], errors="coerce")
    finite_y = finite_y[np.isfinite(finite_y)]
    use_y_clip = st.checkbox("Hide points outside Y range", value=False)
    if use_y_clip and not finite_y.empty:
        y_min = float(finite_y.min())
        y_max = float(finite_y.max())
        if y_max > y_min:
            clip_min, clip_max = st.slider(
                "Y visibility range",
                min_value=y_min,
                max_value=y_max,
                value=(y_min, y_max),
            )
            corrected_df = corrected_df[
                (pd.to_numeric(corrected_df[y_col], errors="coerce") >= clip_min)
                & (pd.to_numeric(corrected_df[y_col], errors="coerce") <= clip_max)
            ].copy()

    st.session_state["custom_plot_corrected_df"] = corrected_df.copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (original)", f"{len(work_df):,}")
    c2.metric("Rows (corrected)", f"{len(corrected_df):,}")
    c3.metric("Rows hidden", f"{max(len(work_df) - len(corrected_df), 0):,}")

    if corrected_df.empty:
        st.warning("No rows remain after correction filters. Adjust your filters to render a plot.")
        return

    _draw_custom_plot(
        df=corrected_df,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
    )

    preview_candidates = [row_id_col, x_col, y_col, color_col, "strategy", "model_id", "status"]
    preview_cols = [c for c in preview_candidates if c and c in corrected_df.columns]
    preview_cols = list(dict.fromkeys(preview_cols))
    extra_cols = [c for c in corrected_df.columns if c not in preview_cols]
    show_cols = preview_cols + extra_cols[: max(0, 12 - len(preview_cols))]
    show_cols = list(dict.fromkeys(show_cols))

    st.markdown("### Corrected Data Preview")
    st.dataframe(corrected_df[show_cols], width="stretch")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    source_slug = source_choice.lower().replace(" ", "_").replace("-", "_")
    csv_bytes = corrected_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download corrected CSV",
        data=csv_bytes,
        file_name=f"{stamp}_{source_slug}_corrected.csv",
        mime="text/csv",
        key="btn_custom_corrected_csv",
    )
