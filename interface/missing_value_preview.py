import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

_PREVIEW_MAX_ROWS = 3_000


@st.cache_data(show_spinner=False)
def _compute_preview(df, strategy: str, _fitted_imputer=None):
    from data_processing import process_for_viz

    return process_for_viz(df, strategy=strategy, fitted_imputer=_fitted_imputer)


def render_preview(df, default_col=None, fitted_imputer=None):
    try:
        from data_processing import STRATEGIES
    except Exception:
        st.error("Preview helpers not available")
        return

    n_prev = min(len(df), _PREVIEW_MAX_ROWS)
    df_prev = df.iloc[-n_prev:].copy()
    if n_prev < len(df):
        st.caption(
            f"Preview uses the last **{n_prev:,}** rows (out of {len(df):,}) "
            "to keep imputation fast."
        )

    X_before, _, dropped = _compute_preview(df_prev, strategy="fill_mean")
    if dropped:
        st.write("**Dropped columns (if any):**", dropped)

    missing_cols = X_before.columns[X_before.isna().any()].tolist()
    if not missing_cols:
        st.info("No features contain missing values — adjust the missing % sliders in the sidebar.")
        return

    st.subheader("Strategy comparison")
    default_sel = [default_col] if default_col and default_col in missing_cols else missing_cols[:1]
    selected_cols = st.multiselect(
        "Columns to compare",
        options=missing_cols,
        default=default_sel,
        help="Select one or more columns with missing values to view imputation comparisons.",
    )
    if not selected_cols:
        st.caption("Select at least one column above to see the comparison.")
        return

    strategy_colors = {
        "ffill": "red",
        "fill_zero": "blue",
        "fill_mean": "green",
        "window_mean": "orange",
        "predictive_imputer": "gold",
    }

    fast_strategies = ["ffill", "fill_zero", "fill_mean", "window_mean"]
    if fitted_imputer is not None:
        active_strategies = fast_strategies + ["predictive_imputer"]
    else:
        show_predictive = st.checkbox(
            "Include predictive imputer (slower — uses RandomForest, no cache found)",
            value=False,
        )
        active_strategies = fast_strategies + (["predictive_imputer"] if show_predictive else [])

    max_points = 150

    for col_name in selected_cols:
        st.markdown(f"#### Column: `{col_name}`")
        series_before = X_before[col_name]
        n = len(series_before)
        sel_pos = np.linspace(0, n - 1, max_points, dtype=int) if n > max_points else np.arange(n, dtype=int)
        sel_index = X_before.index[sel_pos]
        series_before_sel = series_before.iloc[sel_pos]
        missing_mask_sel = series_before_sel.isna()
        x_vals = np.arange(len(sel_pos))

        all_panels = ["_original"] + active_strategies
        cols_per_row = 3
        for row_start in range(0, len(all_panels), cols_per_row):
            row_panels = all_panels[row_start : row_start + cols_per_row]
            grid = st.columns(len(row_panels))
            for col_ui, panel in zip(grid, row_panels):
                fig, ax = plt.subplots(figsize=(5, 3))
                try:
                    if panel == "_original":
                        ax.plot(x_vals, series_before_sel.values, color="tab:blue", lw=1)
                        if missing_mask_sel.any():
                            mean_val = series_before_sel.mean()
                            ph_y = mean_val if not np.isnan(mean_val) else 0.0
                            ax.scatter(
                                x_vals[missing_mask_sel.values],
                                np.full(missing_mask_sel.sum(), ph_y),
                                color="gray",
                                s=15,
                                marker="x",
                                label="Missing",
                            )
                            ax.legend(fontsize="x-small")
                        ax.set_title(f"Original — {col_name}", fontsize=9)
                    else:
                        _, Xa, _ = _compute_preview(df_prev, strategy=panel, _fitted_imputer=fitted_imputer)
                        if col_name not in Xa.columns:
                            ax.text(
                                0.5,
                                0.5,
                                "Column dropped",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                                fontsize=8,
                            )
                        else:
                            aligned = pd.Series(index=X_before.index, dtype=float)
                            aligned.loc[Xa.index] = Xa[col_name].values
                            aligned_sel = aligned.loc[sel_index]
                            ax.plot(x_vals, aligned_sel.values, color="tab:blue", lw=1)
                            imputed_vals = aligned_sel[missing_mask_sel]
                            if not imputed_vals.dropna().empty:
                                ax.scatter(
                                    x_vals[missing_mask_sel.values],
                                    imputed_vals.values,
                                    color=strategy_colors.get(panel, "purple"),
                                    s=15,
                                    label="Imputed",
                                )
                                ax.legend(fontsize="x-small")
                        ax.set_title(panel, fontsize=9)
                except Exception as exc:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error:\n{exc}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=7,
                        wrap=True,
                    )

                ax.set_ylabel("Value", fontsize=8)
                ax.set_xlabel("Sampled row", fontsize=8)
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                col_ui.pyplot(fig, width="stretch")
                plt.close(fig)


def render_missing_value_preview_tab(df_with_missing, target_col, fitted_imputer=None):
    render_preview(df_with_missing, default_col=target_col, fitted_imputer=fitted_imputer)
