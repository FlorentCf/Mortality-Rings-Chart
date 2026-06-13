from __future__ import annotations

import calendar
import json
import math
import subprocess
import urllib.request
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import generate_belgium_mortality_rings_chart_only as chart_only
import generate_belgium_mortality_rings_final as final


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTEXT_DIR = OUTPUT_DIR / "context_sources"
MP4_PATH = OUTPUT_DIR / "belgium_mortality_rings_context_color_4x3_1992_2025.mp4"
X_MP4_PATH = OUTPUT_DIR / "belgium_mortality_rings_context_color_4x3_1992_2025_x_h264_aac.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "belgium_mortality_rings_context_color_4x3_1992_2025_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "belgium_mortality_rings_context_color_4x3_1992_2025_contact_sheet.png"
WEEKLY_CONTEXT_CSV = OUTPUT_DIR / "belgium_weekly_context_scores_1992_2025.csv"

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=50.85&longitude=4.35"
    "&start_date=1992-01-01&end_date=2025-12-31"
    "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
    "&timezone=Europe%2FBrussels"
)
OWID_COVID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
ECDC_ILI_ARI_URL = "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/ILIARIRates.csv"
ECDC_SARI_URL = "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/SARIRates.csv"
ECDC_ACTIVITY_URL = "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/activityFluTypeSubtype.csv"
STATBEL_COD_URL = "https://statbel.fgov.be/sites/default/files/files/opendata/COD/opendata_COD_cause.zip"

CONTEXT_PALETTE = [
    "#fffaf0",
    "#f5dfb6",
    "#dfb368",
    "#be7b3c",
    "#914c25",
    "#64291a",
    "#431414",
    "#23090b",
    "#070203",
]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    clean = value.strip().lstrip("#")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


PALETTE_VALUES = np.linspace(0.0, 1.0, len(CONTEXT_PALETTE))
PALETTE_COLORS = np.array([hex_to_rgb(color) for color in CONTEXT_PALETTE], dtype=float)


def context_colors(score: np.ndarray) -> np.ndarray:
    clamped = np.clip(score, 0.0, 1.0)
    r = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 0])
    g = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 1])
    b = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 2])
    return np.column_stack((r, g, b)).astype(np.float32)


def years() -> list[int]:
    return final.MODEL.years


def blank_matrix() -> np.ndarray:
    return np.zeros((len(years()), final.WEEK_COUNT), dtype=float)


def date_to_model_slot(date: pd.Timestamp) -> tuple[int, int] | None:
    year = int(date.year)
    if year not in final.MODEL.years:
        return None
    day_of_year = int(date.dayofyear) - 1
    days_in_year = 366 if calendar.isleap(year) else 365
    return final.MODEL.years.index(year), final.base.week_index_for_day(day_of_year, days_in_year)


def aggregate_daily_score(df: pd.DataFrame, score_col: str) -> np.ndarray:
    sums = blank_matrix()
    counts = blank_matrix()
    maxes = blank_matrix()
    for row in df[["date", score_col]].itertuples(index=False):
        value = getattr(row, score_col)
        if pd.isna(value):
            continue
        slot = date_to_model_slot(pd.Timestamp(row.date))
        if slot is None:
            continue
        year_index, week_index = slot
        score = float(np.clip(value, 0.0, 1.0))
        sums[year_index, week_index] += score
        counts[year_index, week_index] += 1.0
        maxes[year_index, week_index] = max(maxes[year_index, week_index], score)
    means = np.divide(sums, np.maximum(1.0, counts), out=np.zeros_like(sums), where=counts > 0)
    return np.maximum(means, maxes * 0.72)


def aggregate_week_score(yearweek_score: pd.DataFrame, score_col: str) -> np.ndarray:
    matrix = blank_matrix()
    for row in yearweek_score[["yearweek", score_col]].itertuples(index=False):
        try:
            year_text, week_text = str(row.yearweek).split("-W")
            iso_year = int(year_text)
            iso_week = int(week_text)
            week_date = pd.Timestamp(date.fromisocalendar(iso_year, iso_week, 4))
        except Exception:
            continue
        slot = date_to_model_slot(week_date)
        if slot is None:
            continue
        value = getattr(row, score_col)
        matrix[slot] = max(matrix[slot], float(np.clip(value, 0.0, 1.0)))
    return matrix


def load_json_cache(path: Path, url: str) -> dict | None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        except Exception as exc:
            print(f"warning: could not download {url}: {exc}")
            return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_cache(path: Path, url: str, **kwargs) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(url, **kwargs)
    except Exception as exc:
        print(f"warning: could not download {url}: {exc}")
        return None
    df.to_csv(path, index=False)
    return df


def percentile_scale(series: pd.Series, percentile: float = 0.95) -> pd.Series:
    positive = series[series > 0].dropna()
    if positive.empty:
        return pd.Series(np.zeros(len(series)), index=series.index)
    denom = max(positive.quantile(percentile), 1e-9)
    return (series / denom).clip(0, 1).fillna(0)


def weather_component() -> np.ndarray:
    data = load_json_cache(CONTEXT_DIR / "open_meteo_brussels_1992_2025.json", OPEN_METEO_URL)
    if data is None or "daily" not in data:
        return blank_matrix()
    daily = pd.DataFrame(data["daily"])
    daily["date"] = pd.to_datetime(daily["time"])
    daily["md"] = daily["date"].dt.strftime("%m-%d")
    baseline = daily[daily["date"].dt.year <= 2019].copy()
    climatology = (
        baseline.groupby("md")
        .agg(
            p10_min=("temperature_2m_min", lambda x: np.nanpercentile(x, 10)),
            p90_max=("temperature_2m_max", lambda x: np.nanpercentile(x, 90)),
        )
        .reset_index()
    )
    daily = daily.merge(climatology, on="md", how="left")
    daily["p10_min"] = daily["p10_min"].ffill().bfill()
    daily["p90_max"] = daily["p90_max"].ffill().bfill()

    heat = ((daily["temperature_2m_max"] - daily["p90_max"]) / 7.0).clip(lower=0)
    cold = ((daily["p10_min"] - daily["temperature_2m_min"]) / 7.0).clip(lower=0)
    daily["weather_score"] = np.maximum(heat, cold).clip(0, 1)
    return aggregate_daily_score(daily, "weather_score")


def covid_component() -> np.ndarray:
    cache = CONTEXT_DIR / "owid_covid_belgium.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        cols = [
            "iso_code",
            "date",
            "new_cases_smoothed_per_million",
            "hosp_patients_per_million",
            "icu_patients_per_million",
        ]
        df = read_csv_cache(cache, OWID_COVID_URL, usecols=lambda column: column in cols)
        if df is None:
            return blank_matrix()
        df = df[df["iso_code"].eq("BEL")].copy()
        df.to_csv(cache, index=False)
    df["date"] = pd.to_datetime(df["date"])
    candidates = []
    for column in ["new_cases_smoothed_per_million", "hosp_patients_per_million", "icu_patients_per_million"]:
        if column in df.columns:
            candidates.append(percentile_scale(df[column].fillna(0), 0.97))
    df["covid_score"] = np.maximum.reduce(candidates) if candidates else 0.0
    return aggregate_daily_score(df, "covid_score")


def ecdc_component() -> np.ndarray:
    rates = []
    ili_cache = CONTEXT_DIR / "ecdc_ili_ari_rates.csv"
    sari_cache = CONTEXT_DIR / "ecdc_sari_rates.csv"
    activity_cache = CONTEXT_DIR / "ecdc_activity_flu_rsv.csv"

    ili = read_csv_cache(ili_cache, ECDC_ILI_ARI_URL)
    sari = read_csv_cache(sari_cache, ECDC_SARI_URL)
    activity = read_csv_cache(activity_cache, ECDC_ACTIVITY_URL)

    if ili is None or sari is None or activity is None:
        return blank_matrix()

    ili = ili[ili["countryname"].eq("Belgium") & ili["age"].eq("total")]
    for indicator in ["ILIconsultationrate", "ARIconsultationrate"]:
        subset = ili[ili["indicator"].eq(indicator)].copy()
        if not subset.empty:
            subset["score"] = percentile_scale(subset["value"].astype(float), 0.95)
            rates.append(aggregate_week_score(subset, "score"))

    sari = sari[sari["countryname"].eq("Belgium") & sari["age"].eq("total")]
    if not sari.empty:
        sari["score"] = percentile_scale(sari["value"].astype(float), 0.95)
        rates.append(aggregate_week_score(sari, "score"))

    activity = activity[
        activity["countryname"].eq("Belgium")
        & activity["age"].eq("total")
        & activity["pathogen"].isin(["Influenza", "RSV"])
    ]
    if not activity.empty:
        grouped = activity.groupby("yearweek", as_index=False)["value"].sum()
        grouped["score"] = percentile_scale(grouped["value"].astype(float), 0.95)
        rates.append(aggregate_week_score(grouped, "score"))

    if not rates:
        return blank_matrix()
    return np.maximum.reduce(rates)


def respiratory_component() -> np.ndarray:
    return np.maximum(covid_component(), ecdc_component())


def cause_component() -> np.ndarray:
    cache = CONTEXT_DIR / "statbel_cause_of_death.zip"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        fallback = OUTPUT_DIR / "statbel" / "opendata_COD_cause.zip"
        if fallback.exists():
            cache.write_bytes(fallback.read_bytes())
        else:
            try:
                urllib.request.urlretrieve(STATBEL_COD_URL, cache)
            except Exception as exc:
                print(f"warning: could not download {STATBEL_COD_URL}: {exc}")
                return blank_matrix()
    with ZipFile(cache) as archive:
        with archive.open(archive.namelist()[0]) as file:
            df = pd.read_csv(file, sep="|")

    monthly = df.groupby(["YYYY", "MM", "CHAP_COD"], as_index=False)["TOTAL"].sum()
    total = monthly.groupby(["YYYY", "MM"], as_index=False)["TOTAL"].sum().rename(columns={"TOTAL": "total"})
    focus = monthly[
        monthly["CHAP_COD"].isin(["A00-B99", "J00-J99", "U071-U072", "I00-I99"])
    ].copy()
    focus["weight"] = np.where(focus["CHAP_COD"].eq("I00-I99"), 0.35, 1.0)
    focus["weighted_total"] = focus["TOTAL"] * focus["weight"]
    focus_total = (
        focus.groupby(["YYYY", "MM"], as_index=False)["weighted_total"].sum().rename(columns={"weighted_total": "focus"})
    )
    merged = total.merge(focus_total, on=["YYYY", "MM"], how="left").fillna({"focus": 0.0})
    merged["share"] = merged["focus"] / merged["total"].clip(lower=1)

    baseline = (
        merged[merged["YYYY"].between(2009, 2019)]
        .groupby("MM")["share"]
        .median()
        .rename("baseline_share")
        .reset_index()
    )
    merged = merged.merge(baseline, on="MM", how="left")
    merged["cause_score"] = ((merged["share"] - merged["baseline_share"]) / 0.085).clip(0, 1).fillna(0)

    daily_rows = []
    for row in merged.itertuples(index=False):
        start = pd.Timestamp(int(row.YYYY), int(row.MM), 1)
        days = calendar.monthrange(int(row.YYYY), int(row.MM))[1]
        for offset in range(days):
            daily_rows.append((start + pd.Timedelta(days=offset), float(row.cause_score)))
    daily = pd.DataFrame(daily_rows, columns=["date", "cause_score"])
    return aggregate_daily_score(daily, "cause_score")


def count_only_boundaries() -> list[np.ndarray]:
    annual_totals = np.array([sum(week.deaths for week in weeks) for weeks in final.MODEL.weeks_by_year], dtype=float)
    mean_widths = (annual_totals / annual_totals.mean()) ** 0.35
    mean_widths = mean_widths / mean_widths.sum() * (final.base.MAX_RADIUS - final.base.PITH_RADIUS)
    boundaries = [np.full(final.base.ANGLE_SAMPLES, final.base.PITH_RADIUS, dtype=float)]

    for year_index, year in enumerate(final.MODEL.years):
        days_in_year = 366 if calendar.isleap(year) else 365
        daily_factor = np.ones(days_in_year, dtype=float)
        weeks = final.MODEL.weeks_by_year[year_index]
        week_rates = np.array([week.deaths / max(1, week.days) * 7.0 for week in weeks], dtype=float)
        week_mean = float(np.mean(week_rates))
        for week in weeks:
            week_rate = week.deaths / max(1, week.days) * 7.0
            factor = 1.0 + 0.62 * np.tanh((week_rate / max(1.0, week_mean) - 1.0) / 0.22)
            daily_factor[week.start_day : week.end_day + 1] = factor

        daily_factor = final.base.circular_gaussian(daily_factor, sigma=2.0)
        daily_factor = np.clip(daily_factor, 0.50, 1.90)
        daily_factor /= daily_factor.mean()

        local = np.zeros(final.base.ANGLE_SAMPLES, dtype=float)
        for sample_index in range(final.base.ANGLE_SAMPLES):
            day_position = sample_index / final.base.ANGLE_SAMPLES * days_in_year
            left = int(math.floor(day_position)) % days_in_year
            right = (left + 1) % days_in_year
            frac = day_position - math.floor(day_position)
            local[sample_index] = daily_factor[left] * (1.0 - frac) + daily_factor[right] * frac
        local = final.base.circular_gaussian(local, sigma=2.6)
        local /= local.mean()
        boundaries.append(boundaries[-1] + mean_widths[year_index] * local)
    return boundaries


def build_context() -> tuple[np.ndarray, pd.DataFrame]:
    print("loading weather context")
    weather = weather_component()
    print("loading respiratory context")
    respiratory = respiratory_component()
    print("loading cause-of-death context")
    cause = cause_component()
    composite = np.maximum.reduce([weather * 0.90, respiratory, cause * 0.92])
    composite = np.clip(composite, 0, 1)

    rows = []
    for year_index, year in enumerate(final.MODEL.years):
        for week_index, week in enumerate(final.MODEL.weeks_by_year[year_index]):
            rows.append(
                {
                    "year": year,
                    "week_index": week_index + 1,
                    "start_day": week.start_day + 1,
                    "end_day": week.end_day + 1,
                    "deaths": week.deaths,
                    "weather_score": round(float(weather[year_index, week_index]), 6),
                    "respiratory_score": round(float(respiratory[year_index, week_index]), 6),
                    "cause_mix_score": round(float(cause[year_index, week_index]), 6),
                    "context_score": round(float(composite[year_index, week_index]), 6),
                    "dominant_context": ["weather", "respiratory", "cause_mix"][
                        int(np.argmax([weather[year_index, week_index] * 0.90, respiratory[year_index, week_index], cause[year_index, week_index] * 0.92]))
                    ],
                }
            )
    context_df = pd.DataFrame(rows)
    context_df.to_csv(WEEKLY_CONTEXT_CSV, index=False)
    return composite, context_df


def generate_context_frame_chunks(context: np.ndarray) -> tuple[list[list[final.RectChunk]], np.ndarray]:
    chunks: list[list[final.RectChunk]] = [[] for _ in range(final.DRAW_FRAME_COUNT)]
    frame_counts = np.zeros(final.DRAW_FRAME_COUNT, dtype=np.int64)
    year_count = len(final.MODEL.years)

    for year_index, days in enumerate(final.MODEL.days_by_year):
        weekly_score = final.base.circular_gaussian(context[year_index], sigma=0.62)
        for day in days:
            n = day.deaths
            rng = np.random.default_rng(final.SEED + day.year * 10_000 + day.day_of_year)
            day_fraction = (day.day_of_year + 0.5) / day.days_in_year
            theta_center = math.tau * day_fraction
            theta = theta_center + rng.normal(0.0, math.tau * rng.uniform(0.28, 0.62) / day.days_in_year, n)
            radial_t = rng.random(n)
            theta += (radial_t - 0.5) * rng.normal(0.0, math.tau * 0.50 / day.days_in_year)

            radius = final.radius_at(year_index, radial_t, theta) + rng.normal(0.0, 0.08, n)
            visual_angle = theta + final.base.ANGLE_OFFSET
            x = np.rint(final.TREE_CENTER[0] + radius * np.cos(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)
            y = np.rint(final.TREE_CENTER[1] + radius * np.sin(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)

            week_float = (theta % math.tau) / math.tau * final.WEEK_COUNT
            score = final.base.circular_interp(weekly_score, week_float)
            base_color = context_colors(score)
            rgb = np.clip(base_color + rng.normal(0.0, 2.4, (n, 3)), 0, 255).astype(np.uint8)
            tx, ty = final.oriented_pixel_offsets(visual_angle)

            radial_progress = np.clip(radial_t + rng.normal(0.0, 0.014, n), 0.0, 0.998)
            progress = (year_index + radial_progress) / year_count
            birth_frames = np.clip((progress * (final.DRAW_FRAME_COUNT - 1)).astype(np.int32), 0, final.DRAW_FRAME_COUNT - 1)

            for frame_index in np.unique(birth_frames):
                mask = birth_frames == frame_index
                chunks[int(frame_index)].append(
                    final.RectChunk(
                        x=x[mask],
                        y=y[mask],
                        rgb=rgb[mask],
                        tx=tx[mask],
                        ty=ty[mask],
                    )
                )
                frame_counts[int(frame_index)] += int(mask.sum())
        if year_index % 5 == 0:
            print(f"prepared context-colored cells through {final.MODEL.years[year_index]}")
    return chunks, frame_counts


def encode_x_compatible() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(MP4_PATH),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(X_MP4_PATH),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)


def inspect_video(path: Path) -> list[str]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], text=True, capture_output=True, check=False)
    return [
        line.strip()
        for line in completed.stderr.splitlines()
        if "Duration:" in line or "Video:" in line or "Audio:" in line
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.MODEL.boundaries = count_only_boundaries()
    context, context_df = build_context()
    chart_only.MP4_PATH = MP4_PATH
    chart_only.X_MP4_PATH = X_MP4_PATH
    chart_only.FINAL_FRAME_PATH = FINAL_FRAME_PATH
    chart_only.CONTACT_SHEET_PATH = CONTACT_SHEET_PATH
    print(f"years: {final.MODEL.years[0]}-{final.MODEL.years[-1]} ({len(final.MODEL.years)} rings)")
    print(f"total cells: {final.MODEL.total_deaths:,}")
    print("geometry: thickness uses death counts only; color uses weather/respiratory/cause-context composite")
    print(context_df.groupby("dominant_context").size().to_string())
    chunks, frame_counts = generate_context_frame_chunks(context)
    final_frame, contact_frames = chart_only.render_all_frames(chunks, frame_counts)
    final_frame.save(FINAL_FRAME_PATH)
    chart_only.save_contact_sheet(contact_frames)
    encode_x_compatible()
    print(f"weekly context: {WEEKLY_CONTEXT_CSV}")
    print(f"video: {MP4_PATH} ({MP4_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"x video: {X_MP4_PATH} ({X_MP4_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"final frame: {FINAL_FRAME_PATH}")
    print(f"contact sheet: {CONTACT_SHEET_PATH}")
    for line in inspect_video(X_MP4_PATH):
        print(f"  {line}")


if __name__ == "__main__":
    main()
