from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    aliases: tuple[str, ...]
    color: str
    higher_is_better: bool = True
    ratio_like: bool = False


METRIC_SPECS: dict[str, MetricSpec] = {
    "Answer Correctness": MetricSpec(
        aliases=(
            "answer correctness",
            "a1",
            "a1 correctness",
            "correctness",
            "answer_correctness",
            "answer correctness / a1",
        ),
        color="#0f8b8d",
    ),
    "Coverage of Key Points": MetricSpec(
        aliases=(
            "coverage of key points",
            "a2",
            "a2 coverage",
            "coverage",
            "key point coverage",
            "key points",
            "coverage of key points / a2",
        ),
        color="#2563eb",
    ),
    "Groundedness / Faithfulness": MetricSpec(
        aliases=(
            "groundedness",
            "faithfulness",
            "a3",
            "a3 groundedness",
            "a3 faithfulness",
            "groundedness / faithfulness",
            "faithfulness / groundedness",
            "groundedness / faithfulness / a3",
            "helpfulness",
            "a3 helpfulness",
            "perceived helpfulness",
            "a3 perceived helpfulness",
        ),
        color="#f4b400",
    ),
    "Unsupported Claims": MetricSpec(
        aliases=(
            "unsupported claims",
            "a4",
            "a4 unsupported claims",
            "hallucinations",
            "unsupported",
            "unsupported_claims",
            "unsupported claims / a4",
        ),
        color="#d9485f",
        higher_is_better=False,
        ratio_like=True,
    ),
    "Precision@1": MetricSpec(
        aliases=(
            "precision@1",
            "p@1",
            "precision 1",
            "precision_at_1",
            "precision at 1",
        ),
        color="#7c4dff",
        ratio_like=True,
    ),
    "Precision@3": MetricSpec(
        aliases=(
            "precision@3",
            "p@3",
            "precision 3",
            "precision_at_3",
            "precision at 3",
        ),
        color="#2a9d8f",
        ratio_like=True,
    ),
    "Precision@6": MetricSpec(
        aliases=(
            "precision@6",
            "p@6",
            "precision 6",
            "precision_at_6",
            "precision at 6",
        ),
        color="#5b7c99",
        ratio_like=True,
    ),
}

QUERY_COL_CANDIDATES = (
    "query",
    "query id",
    "query_id",
    "query name",
    "question",
    "question text",
    "prompt",
    "user question",
)

TYPE_COL_CANDIDATES = (
    "type",
    "query type",
    "query_type",
    "category",
    "question type",
    "guideline",
    "section",
)

SUMMARY_METRIC_COL_CANDIDATES = (
    "metric",
    "metric name",
    "measure",
    "evaluation metric",
)

SUMMARY_VALUE_COL_CANDIDATES = (
    "value",
    "score",
    "mean",
    "average",
    "result",
)

SUMMARY_STD_COL_CANDIDATES = (
    "std",
    "stdev",
    "sd",
    "standard deviation",
    "error",
)

STYLE = {
    "figure.facecolor": "#f5efe6",
    "axes.facecolor": "#fffdf9",
    "savefig.facecolor": "#f5efe6",
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

GRID_COLOR = "#dbcdbb"
TEXT_COLOR = "#2e2923"
MUTED_TEXT = "#756a5d"
AXIS_COLOR = "#bcae9d"
PANEL_COLOR = "#fffdf9"

FRACTION_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*/\s*(-?\d+(?:[.,]\d+)?)")
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
PLUS_MINUS_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(?:±|\+/-)\s*(-?\d+(?:[.,]\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate readable plots for evaluation metrics such as A1/A2/A3/A4 "
            "and Precision@k from a detailed table or a compact summary sheet."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the evaluation sheet or table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to <input_stem>_plots next to the input file.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name. If omitted, all sheets are combined.",
    )
    parser.add_argument(
        "--title",
        default="Evaluation Metrics Dashboard",
        help="Title prefix used in the combined plots.",
    )
    parser.add_argument(
        "--include-mcq",
        action="store_true",
        help="Keep rows tagged as MCQ if a type/category column exists.",
    )
    return parser.parse_args()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9@]+", " ", str(text).strip().lower()).strip()


def find_best_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {column: norm(column) for column in columns}

    for candidate in candidates:
        candidate_norm = norm(candidate)
        for original, current in normalized.items():
            if current == candidate_norm:
                return original

    for candidate in candidates:
        candidate_norm = norm(candidate)
        for original, current in normalized.items():
            if candidate_norm in current:
                return original

    return None


def detect_metric_columns(columns: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    remaining = list(columns)

    for final_name, spec in METRIC_SPECS.items():
        aliases = (final_name, *spec.aliases)
        for alias in aliases:
            alias_norm = norm(alias)
            match = None

            for column in remaining:
                if norm(column) == alias_norm:
                    match = column
                    break

            if match is None:
                for column in remaining:
                    if alias_norm in norm(column):
                        match = column
                        break

            if match is not None:
                found[final_name] = match
                remaining.remove(match)
                break

    return found


def canonicalize_metric_name(label: object) -> str | None:
    target = norm(label)
    if not target:
        return None

    for final_name, spec in METRIC_SPECS.items():
        for candidate in (final_name, *spec.aliases):
            candidate_norm = norm(candidate)
            if candidate_norm == target or candidate_norm in target or target in candidate_norm:
                return final_name

    return None


def read_input_table(path: Path, sheet: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        if sheet:
            return pd.read_excel(path, sheet_name=sheet)
        return read_all_sheets(path)
    if suffix == ".json":
        return read_json_records(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def read_all_sheets(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    parts: list[pd.DataFrame] = []

    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        if frame.empty:
            continue
        frame["__sheet__"] = sheet_name
        parts.append(frame)

    if not parts:
        raise ValueError("No readable sheets found in the workbook.")

    return pd.concat(parts, ignore_index=True)


def read_json_records(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        for key in ("rows", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)

    raise ValueError(
        "JSON input must be a list of records or contain a top-level rows/data/results list."
    )


def wrap_label(value: object, width: int = 46, max_lines: int = 3) -> str:
    text = str(value).strip()
    wrapped = textwrap.wrap(text, width=width) or [text]
    if len(wrapped) > max_lines:
        wrapped = wrapped[: max_lines - 1] + [wrapped[max_lines - 1] + "..."]
    return "\n".join(wrapped)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9@._-]+", "_", value).strip("_")


def metric_spec(metric_name: str) -> MetricSpec:
    return METRIC_SPECS[metric_name]


def parse_metric_value(value: object, metric_name: str) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return math.nan

    lowered = text.lower()
    if lowered in {"na", "n/a", "none", "null", "nan", "-"}:
        return math.nan

    normalized_text = text.replace(",", ".")
    spec = metric_spec(metric_name)

    if "%" in normalized_text:
        match = NUMBER_RE.search(normalized_text)
        if match:
            return float(match.group()) / 100.0

    if spec.ratio_like:
        fraction_match = FRACTION_RE.search(normalized_text)
        if fraction_match:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator == 0:
                return math.nan
            return numerator / denominator

    direct = pd.to_numeric(normalized_text, errors="coerce")
    if not pd.isna(direct):
        return float(direct)

    match = NUMBER_RE.search(normalized_text)
    if match:
        return float(match.group())

    return math.nan


def parse_metric_entry(value: object, metric_name: str) -> tuple[float, float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (math.nan, math.nan)

    text = str(value).strip()
    if not text:
        return (math.nan, math.nan)

    normalized_text = text.replace(",", ".")
    plus_minus_match = PLUS_MINUS_RE.search(normalized_text)
    if plus_minus_match:
        mean_value = float(plus_minus_match.group(1))
        std_value = abs(float(plus_minus_match.group(2)))
        return (mean_value, std_value)

    return (parse_metric_value(value, metric_name), math.nan)


def infer_metric_bounds(metric_name: str, values: pd.Series) -> tuple[float, float, str]:
    clean = values.dropna()
    if clean.empty:
        return (0.0, 1.0, "0-1")

    minimum = float(clean.min())
    maximum = float(clean.max())

    if minimum >= 0 and maximum <= 1.05:
        return (0.0, 1.0, "0-1")
    if minimum >= 0 and maximum <= 5.05:
        return (0.0, 5.0, "0-5")
    if minimum >= 0 and maximum <= 10.05:
        return (0.0, 10.0, "0-10")
    if minimum >= 0 and maximum <= 100.5:
        return (0.0, 100.0, "0-100")

    padding = 0.08 * (maximum - minimum if maximum != minimum else 1.0)
    return (minimum - padding, maximum + padding, "observed")


def normalize_for_comparison(values: pd.Series, metric_name: str) -> pd.Series:
    lower, upper, scale_label = infer_metric_bounds(metric_name, values)
    clean = values.astype(float)

    if scale_label in {"0-1", "0-5", "0-10", "0-100"} and upper > lower:
        scaled = (clean - lower) / (upper - lower)
    else:
        actual_min = clean.min()
        actual_max = clean.max()
        if pd.isna(actual_min) or pd.isna(actual_max) or actual_min == actual_max:
            scaled = pd.Series(0.5, index=clean.index, dtype=float)
        else:
            scaled = (clean - actual_min) / (actual_max - actual_min)

    scaled = scaled.clip(lower=0.0, upper=1.0)
    if not metric_spec(metric_name).higher_is_better:
        scaled = 1.0 - scaled
    return scaled


def add_plot_style() -> None:
    plt.rcParams.update(STYLE)


def to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    return (red, green, blue, alpha)


def blend_hex(hex_color: str, target_hex: str, weight: float) -> str:
    source = hex_color.lstrip("#")
    target = target_hex.lstrip("#")
    weight = max(0.0, min(1.0, weight))

    values: list[int] = []
    for start in range(0, 6, 2):
        source_value = int(source[start : start + 2], 16)
        target_value = int(target[start : start + 2], 16)
        blended = round(source_value * (1.0 - weight) + target_value * weight)
        values.append(blended)

    return "#" + "".join(f"{value:02x}" for value in values)


def bar_color_sequence(metric_name: str, values: pd.Series) -> list[tuple[float, float, float, float]]:
    score = normalize_for_comparison(values, metric_name)
    base = metric_spec(metric_name).color
    colors: list[tuple[float, float, float, float]] = []
    for item in score.fillna(0.5):
        alpha = 0.45 + 0.45 * float(item)
        colors.append(to_rgba(base, alpha=alpha))
    return colors


def format_metric_value(metric_name: str, value: float) -> str:
    if pd.isna(value):
        return "NA"

    _, upper, scale_label = infer_metric_bounds(metric_name, pd.Series([value]))
    if scale_label == "0-100" or upper == 100:
        return f"{value:.1f}"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}"


def build_summary_row(
    metric_name: str,
    mean: float,
    *,
    std: float = math.nan,
    count: int = 1,
    min_value: float | None = None,
    max_value: float | None = None,
) -> dict[str, object]:
    if min_value is None or pd.isna(min_value):
        min_value = mean
    if max_value is None or pd.isna(max_value):
        max_value = mean

    lower, upper, scale_label = infer_metric_bounds(metric_name, pd.Series([mean]))
    comparison_mean_pct = float(normalize_for_comparison(pd.Series([mean]), metric_name).iloc[0] * 100.0)
    comparison_std_pct = math.nan
    if not pd.isna(std) and upper > lower:
        comparison_std_pct = abs(float(std)) / (upper - lower) * 100.0

    return {
        "metric": metric_name,
        "count": int(count),
        "mean": float(mean),
        "std": float(std) if not pd.isna(std) else math.nan,
        "min": float(min_value),
        "max": float(max_value),
        "scale_label": scale_label,
        "higher_is_better": metric_spec(metric_name).higher_is_better,
        "comparison_mean_pct": comparison_mean_pct,
        "comparison_std_pct": comparison_std_pct,
        "axis_lower": lower,
        "axis_upper": upper,
    }


def build_summary_table(source: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = source.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    work.columns = [str(column).strip() for column in work.columns]

    rows: list[dict[str, object]] = []
    metric_col = find_best_column(list(work.columns), SUMMARY_METRIC_COL_CANDIDATES)
    value_col = find_best_column(list(work.columns), SUMMARY_VALUE_COL_CANDIDATES)
    std_col = find_best_column(list(work.columns), SUMMARY_STD_COL_CANDIDATES)

    if metric_col is not None and value_col is not None:
        for _, row in work.iterrows():
            metric_name = canonicalize_metric_name(row.get(metric_col))
            if metric_name is None:
                continue

            mean_value, embedded_std = parse_metric_entry(row.get(value_col), metric_name)
            if pd.isna(mean_value):
                continue

            std_value = embedded_std
            if std_col is not None:
                parsed_std = parse_metric_value(row.get(std_col), metric_name)
                if not pd.isna(parsed_std):
                    std_value = parsed_std

            rows.append(build_summary_row(metric_name, mean_value, std=std_value))
    else:
        metric_cols = detect_metric_columns(list(work.columns))
        if not metric_cols:
            raise ValueError(
                "No query column or summary metric columns were detected. "
                "Provide either a detailed table or a compact summary table."
            )

        for metric_name, column_name in metric_cols.items():
            values = work[column_name].dropna()
            if values.empty:
                continue

            parsed = values.apply(lambda item: parse_metric_entry(item, metric_name))
            means = pd.Series([item[0] for item in parsed], dtype=float).dropna()
            if means.empty:
                continue

            embedded_stds = [item[1] for item in parsed if not pd.isna(item[1])]
            std_value = math.nan
            if len(means) > 1:
                std_value = float(means.std(ddof=1))
            elif embedded_stds:
                std_value = float(embedded_stds[0])

            rows.append(
                build_summary_row(
                    metric_name,
                    float(means.mean()),
                    std=std_value,
                    count=int(means.count()),
                    min_value=float(means.min()),
                    max_value=float(means.max()),
                )
            )

    if not rows:
        raise ValueError("No summary metric values could be parsed from the input.")

    summary = pd.DataFrame(rows)
    metric_names = [metric for metric in METRIC_SPECS if metric in summary["metric"].tolist()]
    summary = summary.sort_values("comparison_mean_pct", ascending=False).reset_index(drop=True)
    return summary, metric_names


def build_clean_table(
    source: pd.DataFrame,
    include_mcq: bool,
) -> tuple[pd.DataFrame, list[str], str | None]:
    work = source.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    work.columns = [str(column).strip() for column in work.columns]

    query_col = find_best_column(list(work.columns), QUERY_COL_CANDIDATES)
    if query_col is None:
        raise ValueError(
            "Could not find a query/question column. Expected one close to: "
            + ", ".join(QUERY_COL_CANDIDATES)
        )

    metric_cols = detect_metric_columns(list(work.columns))
    if not metric_cols:
        raise ValueError(
            "No supported metric columns were detected. Rename the columns closer to: "
            + ", ".join(METRIC_SPECS.keys())
        )

    type_col = find_best_column(list(work.columns), TYPE_COL_CANDIDATES)

    keep_cols = [query_col, *metric_cols.values()]
    if type_col is not None:
        keep_cols.append(type_col)

    clean = work[keep_cols].copy()
    rename_map = {query_col: "Query", **{column: metric for metric, column in metric_cols.items()}}
    if type_col is not None:
        rename_map[type_col] = "Group"
    clean = clean.rename(columns=rename_map)

    if "Group" in clean.columns and not include_mcq:
        clean = clean[
            ~clean["Group"].astype(str).str.contains(r"\bmcq\b", case=False, na=False)
        ].copy()

    for metric_name in metric_cols:
        clean[metric_name] = clean[metric_name].apply(lambda value: parse_metric_value(value, metric_name))

    clean["Query"] = clean["Query"].astype(str).str.strip()
    clean = clean[clean["Query"].astype(bool)].copy()

    metric_names = [metric for metric in METRIC_SPECS if metric in clean.columns]
    clean = clean.dropna(subset=metric_names, how="all").copy()

    if clean.empty:
        raise ValueError("No usable rows remain after cleaning.")

    if "Group" in clean.columns:
        clean["Group"] = clean["Group"].astype(str).str.strip()
        if clean["Group"].replace("", np.nan).dropna().empty:
            clean = clean.drop(columns=["Group"])

    ordered_columns = ["Query", *metric_names]
    if "Group" in clean.columns:
        ordered_columns.append("Group")
    clean = clean[ordered_columns]

    return clean, metric_names, "Group" if "Group" in clean.columns else None


def build_metric_summary(clean: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for metric_name in metric_names:
        series = clean[metric_name].dropna().astype(float)
        lower, upper, scale_label = infer_metric_bounds(metric_name, series)
        comparison = normalize_for_comparison(series, metric_name)
        rows.append(
            {
                "metric": metric_name,
                "count": int(series.count()),
                "mean": series.mean(),
                "std": series.std(ddof=1),
                "min": series.min(),
                "max": series.max(),
                "scale_label": scale_label,
                "higher_is_better": metric_spec(metric_name).higher_is_better,
                "comparison_mean_pct": comparison.mean() * 100.0,
                "comparison_std_pct": comparison.std(ddof=1) * 100.0,
                "axis_lower": lower,
                "axis_upper": upper,
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("comparison_mean_pct", ascending=False).reset_index(drop=True)


def plot_metric_bars(clean: pd.DataFrame, metric_name: str, output_dir: Path) -> None:
    summary = (
        clean.groupby("Query", dropna=False)[metric_name]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)

    ascending = not metric_spec(metric_name).higher_is_better
    summary = summary.sort_values("mean", ascending=ascending).reset_index(drop=True)

    lower, upper, _ = infer_metric_bounds(metric_name, summary["mean"])
    figure_height = max(6.0, 0.52 * len(summary) + 2.4)
    fig, ax = plt.subplots(figsize=(13.5, figure_height))

    positions = np.arange(len(summary))
    colors = bar_color_sequence(metric_name, summary["mean"])
    bars = ax.barh(
        positions,
        summary["mean"],
        xerr=summary["std"] if summary["count"].max() > 1 else None,
        color=colors,
        edgecolor="#51473f",
        linewidth=0.9,
        capsize=4,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels([wrap_label(label, width=54) for label in summary["Query"]])
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=(0, (4, 4)), color=GRID_COLOR, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR)

    total_span = upper - lower if upper != lower else 1.0
    right_padding = upper + total_span * 0.18
    ax.set_xlim(lower, right_padding)
    ax.set_xlabel("Mean score", color=TEXT_COLOR, labelpad=10)

    direction_note = "Higher is better" if metric_spec(metric_name).higher_is_better else "Lower is better"
    ax.set_title(f"{metric_name} by Question", loc="left", color=TEXT_COLOR, pad=18, weight="bold")
    ax.text(
        0.0,
        1.02,
        direction_note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED_TEXT,
        fontsize=10,
    )

    overall_mean = float(summary["mean"].mean())
    ax.axvline(
        overall_mean,
        color="#3d405b",
        linestyle=(0, (2, 3)),
        linewidth=1.7,
        alpha=0.85,
    )
    ax.text(
        overall_mean,
        1.01,
        f"Overall mean {format_metric_value(metric_name, overall_mean)}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color="#3d405b",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffdfa", edgecolor="#cabfa8"),
    )

    label_offset = total_span * 0.025
    for bar, (_, row) in zip(bars, summary.iterrows(), strict=False):
        value = float(row["mean"])
        label = format_metric_value(metric_name, value)
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{label}  (n={int(row['count'])})",
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=9,
        )

    for spine in ax.spines.values():
        spine.set_color("#b7ad9d")

    fig.tight_layout()
    fig.savefig(output_dir / f"{sanitize_filename(metric_name)}_by_question.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_overview(
    summary: pd.DataFrame,
    title: str,
    output_dir: Path,
) -> None:
    ordered = summary.sort_values("comparison_mean_pct", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])
    ax.set_facecolor(PANEL_COLOR)

    positions = np.arange(len(ordered))
    bars = ax.barh(
        positions,
        ordered["comparison_mean_pct"],
        color=[metric_spec(metric).color for metric in ordered["metric"]],
        edgecolor=[blend_hex(metric_spec(metric).color, "#201a16", 0.45) for metric in ordered["metric"]],
        linewidth=1.1,
        alpha=0.95,
        height=0.78,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["metric"])
    ax.set_xlim(0, 108)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(axis="x", linestyle=(0, (4, 4)), color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.set_xlabel("Normalized quality score (0-100)", color=TEXT_COLOR, labelpad=10)
    ax.set_title(title, loc="left", color=TEXT_COLOR, pad=12, weight="bold")
    ax.tick_params(colors=TEXT_COLOR)
    ax.margins(y=0.06)

    for bar, (_, row) in zip(bars, ordered.iterrows(), strict=False):
        metric_name = str(row["metric"])
        metric_color = metric_spec(metric_name).color
        value = float(row["comparison_mean_pct"])
        center_y = bar.get_y() + bar.get_height() / 2

        ax.scatter(
            [value],
            [center_y],
            s=62,
            color="#fffdfa",
            edgecolor=blend_hex(metric_color, "#1b1714", 0.45),
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            min(value + 2.0, 104.8),
            center_y,
            f"{value:.0f}%",
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": blend_hex(metric_color, "#fffaf4", 0.86),
                "edgecolor": blend_hex(metric_color, "#231d18", 0.52),
                "linewidth": 0.9,
            },
        )

    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)

    fig.tight_layout()
    fig.savefig(output_dir / "metric_overview_normalized.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_distributions(clean: pd.DataFrame, metric_names: list[str], output_dir: Path) -> None:
    values = [normalize_for_comparison(clean[metric].dropna(), metric) * 100.0 for metric in metric_names]
    labels = metric_names

    fig, ax = plt.subplots(figsize=(12.0, max(5.6, 0.75 * len(metric_names) + 2.2)))
    boxplot_kwargs = {
        "x": values,
        "vert": False,
        "patch_artist": True,
        "showfliers": False,
        "widths": 0.58,
        "medianprops": {"color": "#2f2a24", "linewidth": 1.6},
        "whiskerprops": {"color": "#7b6f62", "linewidth": 1.2},
        "capprops": {"color": "#7b6f62", "linewidth": 1.2},
    }
    try:
        box = ax.boxplot(tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        box = ax.boxplot(labels=labels, **boxplot_kwargs)

    for patch, metric_name in zip(box["boxes"], metric_names, strict=False):
        patch.set_facecolor(metric_spec(metric_name).color)
        patch.set_edgecolor("#4f453b")
        patch.set_alpha(0.78)

    rng = np.random.default_rng(42)
    for idx, metric_name in enumerate(metric_names, start=1):
        series = normalize_for_comparison(clean[metric_name].dropna(), metric_name) * 100.0
        if series.empty:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(series))
        ax.scatter(
            series,
            np.full(len(series), idx) + jitter,
            s=24,
            alpha=0.55,
            color="#fffdfa",
            edgecolor=metric_spec(metric_name).color,
            linewidth=0.9,
        )

    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle=(0, (4, 4)), color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.set_xlabel("Normalized quality score (0-100)", color=TEXT_COLOR, labelpad=10)
    ax.set_title("Metric Distributions", loc="left", color=TEXT_COLOR, pad=16, weight="bold")
    ax.text(
        0.0,
        1.02,
        "Scores are normalized per metric scale; lower-is-better metrics are inverted.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED_TEXT,
        fontsize=10,
    )

    for spine in ax.spines.values():
        spine.set_color("#b7ad9d")

    fig.tight_layout()
    fig.savefig(output_dir / "metric_distributions_normalized.png", bbox_inches="tight")
    plt.close(fig)


def plot_query_heatmap(clean: pd.DataFrame, metric_names: list[str], output_dir: Path) -> None:
    raw = clean.groupby("Query", dropna=False)[metric_names].mean(numeric_only=True)
    normalized = pd.DataFrame(index=raw.index)
    for metric_name in metric_names:
        normalized[metric_name] = normalize_for_comparison(raw[metric_name], metric_name) * 100.0

    normalized["__sort__"] = normalized.mean(axis=1, skipna=True)
    normalized = normalized.sort_values("__sort__", ascending=False).drop(columns="__sort__")
    raw = raw.loc[normalized.index]

    fig_height = max(5.8, 0.55 * len(normalized) + 2.0)
    fig_width = max(10.0, 1.22 * len(metric_names) + 5.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(normalized.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_yticklabels([wrap_label(label, width=38, max_lines=2) for label in normalized.index])
    ax.set_title("Question x Metric Heatmap", loc="left", color=TEXT_COLOR, pad=16, weight="bold")

    for row_index in range(normalized.shape[0]):
        for column_index, metric_name in enumerate(metric_names):
            raw_value = raw.iloc[row_index, column_index]
            if pd.isna(raw_value):
                label = "NA"
            else:
                label = format_metric_value(metric_name, float(raw_value))
            normalized_value = normalized.iloc[row_index, column_index]
            text_color = "#132238" if normalized_value < 65 else "#fffdfa"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.8,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    colorbar.set_label("Normalized quality score", color=TEXT_COLOR)
    colorbar.outline.set_edgecolor("#b7ad9d")

    for spine in ax.spines.values():
        spine.set_color("#b7ad9d")

    fig.tight_layout()
    fig.savefig(output_dir / "question_metric_heatmap_normalized.png", bbox_inches="tight")
    plt.close(fig)


def plot_group_heatmap(
    clean: pd.DataFrame,
    group_column: str,
    metric_names: list[str],
    output_dir: Path,
) -> None:
    grouped = clean.groupby(group_column, dropna=False)[metric_names].mean(numeric_only=True)
    if grouped.shape[0] <= 1:
        return

    normalized = pd.DataFrame(index=grouped.index)
    for metric_name in metric_names:
        normalized[metric_name] = normalize_for_comparison(grouped[metric_name], metric_name) * 100.0

    fig_height = max(4.6, 0.8 * len(grouped.index) + 2.0)
    fig_width = max(9.5, 1.25 * len(metric_names) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(normalized.values, aspect="auto", cmap="OrRd", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(grouped.index)))
    ax.set_yticklabels([wrap_label(label, width=26, max_lines=2) for label in grouped.index])
    ax.set_title("Metric Breakdown by Group", loc="left", color=TEXT_COLOR, pad=16, weight="bold")

    for row_index in range(grouped.shape[0]):
        for column_index, metric_name in enumerate(metric_names):
            raw_value = grouped.iloc[row_index, column_index]
            label = "NA" if pd.isna(raw_value) else format_metric_value(metric_name, float(raw_value))
            color = "#2f2a24" if normalized.iloc[row_index, column_index] < 60 else "#fffdfa"
            ax.text(column_index, row_index, label, ha="center", va="center", fontsize=9, color=color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Normalized quality score", color=TEXT_COLOR)
    colorbar.outline.set_edgecolor("#b7ad9d")

    for spine in ax.spines.values():
        spine.set_color("#b7ad9d")

    fig.tight_layout()
    fig.savefig(output_dir / "metric_breakdown_by_group_normalized.png", bbox_inches="tight")
    plt.close(fig)


def plot_summary_small_multiples(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("comparison_mean_pct", ascending=False).reset_index(drop=True)
    column_count = 2 if len(ordered) > 4 else 1
    row_count = math.ceil(len(ordered) / column_count)
    fig_width = 13.5 if column_count == 2 else 8.4
    fig_height = max(3.8, row_count * 2.8)

    fig, axes = plt.subplots(row_count, column_count, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    for ax, (_, row) in zip(axes.flat, ordered.iterrows(), strict=False):
        metric_name = str(row["metric"])
        metric_color = metric_spec(metric_name).color
        lower = float(row["axis_lower"])
        upper = float(row["axis_upper"])
        mean_value = float(row["mean"])
        std_value = float(row["std"]) if not pd.isna(row["std"]) else math.nan
        span = upper - lower if upper != lower else 1.0

        ax.set_facecolor(blend_hex(metric_color, "#fffaf4", 0.92))
        ax.axvspan(lower, upper, color=blend_hex(metric_color, "#ffffff", 0.90), alpha=0.95, zorder=0)
        ax.barh(
            [0],
            [max(mean_value - lower, 0.0)],
            left=lower,
            height=0.46,
            color=metric_color,
            edgecolor=blend_hex(metric_color, "#201a16", 0.45),
            linewidth=1.0,
            alpha=0.92,
            zorder=2,
        )
        ax.scatter(
            [mean_value],
            [0],
            s=54,
            color="#fffdfa",
            edgecolor=blend_hex(metric_color, "#1f1915", 0.45),
            linewidth=1.1,
            zorder=3,
        )

        if not pd.isna(std_value) and std_value > 0:
            ax.errorbar(
                [mean_value],
                [0],
                xerr=[[std_value], [std_value]],
                fmt="none",
                ecolor=blend_hex(metric_color, "#1f1915", 0.52),
                elinewidth=1.3,
                capsize=4,
                zorder=4,
            )

        ax.set_xlim(lower, upper + span * 0.03)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([])
        ax.grid(axis="x", linestyle=(0, (4, 4)), color=GRID_COLOR)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_COLOR)
        ax.set_title(
            wrap_label(metric_name, width=24, max_lines=2),
            loc="left",
            color=TEXT_COLOR,
            pad=10,
            weight="bold",
        )

        detail = format_metric_value(metric_name, mean_value)
        if not pd.isna(std_value) and std_value > 0:
            detail += f" +/- {format_metric_value(metric_name, std_value)}"

        ax.text(
            0.98,
            0.92,
            detail,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=MUTED_TEXT,
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#fffdfa",
                "edgecolor": blend_hex(metric_color, "#ffffff", 0.35),
                "linewidth": 0.9,
            },
        )

        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)

    for ax in axes.flat[len(ordered):]:
        ax.axis("off")

    fig.suptitle(
        "Metric Summary by Native Scale",
        x=0.06,
        y=0.988,
        ha="left",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "metric_summary_native_scales.png", bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    clean: pd.DataFrame,
    summary: pd.DataFrame,
    metric_names: list[str],
    group_column: str | None,
    title: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean.to_csv(output_dir / "cleaned_metrics_used_for_plots.csv", index=False)
    summary.round(4).to_csv(output_dir / "metric_summary_statistics.csv", index=False)

    plot_metric_overview(summary, title, output_dir)
    plot_metric_distributions(clean, metric_names, output_dir)
    plot_query_heatmap(clean, metric_names, output_dir)

    for metric_name in metric_names:
        if clean[metric_name].dropna().empty:
            continue
        plot_metric_bars(clean[["Query", metric_name]], metric_name, output_dir)

    if group_column is not None:
        plot_group_heatmap(clean, group_column, metric_names, output_dir)


def save_summary_outputs(
    summary: pd.DataFrame,
    metric_names: list[str],
    title: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(output_dir / "metric_summary_statistics.csv", index=False)

    plot_metric_overview(summary, title, output_dir)
    plot_summary_small_multiples(summary, output_dir)

    pd.DataFrame({"metric": metric_names}).to_csv(
        output_dir / "metrics_detected.csv",
        index=False,
    )


def main() -> None:
    add_plot_style()
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_dir = args.output_dir or args.input.with_name(f"{args.input.stem}_plots")
    raw = read_input_table(args.input, args.sheet)

    try:
        clean, metric_names, group_column = build_clean_table(raw, include_mcq=args.include_mcq)
    except ValueError:
        summary, metric_names = build_summary_table(raw)
        save_summary_outputs(summary, metric_names, args.title, output_dir)
        print(f"Saved summary plots to: {output_dir}")
        print("Metrics detected: " + ", ".join(metric_names))
        return

    summary = build_metric_summary(clean, metric_names)
    save_outputs(clean, summary, metric_names, group_column, args.title, output_dir)

    print(f"Saved detailed plots to: {output_dir}")
    print("Metrics detected: " + ", ".join(metric_names))


if __name__ == "__main__":
    main()
