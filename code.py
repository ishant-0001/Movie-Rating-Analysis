import numpy as np
import pandas as pd

# ==========================================================
# 1. LOAD RAW DATA & INITIAL AUDIT
# ==========================================================
MISSING_TOKENS = ["", "N/A", "NULL", "None", "?"]

df_raw = pd.read_csv(
    "movies.csv",
    na_values=MISSING_TOKENS,
    keep_default_na=True,
)

print("=== 1. RAW DATA AUDIT ===")
print(f"Total rows: {len(df_raw)} | Columns: {df_raw.shape[1]}")
print(f"Exact duplicate rows: {df_raw.duplicated().sum()}")
print(f"Duplicate movie_ids:  {df_raw['movie_id'].duplicated().sum()}")
print("\nMissing values before cleaning:")
print(df_raw.isna().sum())


# ==========================================================
# 2. VECTORIZED DATA CLEANING (PANDAS + NUMPY ONLY)
# ==========================================================
df = df_raw.copy()

# 2.1 Normalize IDs, titles, directors & drop exact/near-duplicates
df["movie_id"] = df["movie_id"].astype(str).str.strip()
df["title"] = (
    df["title"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .str.title()
)
df["director"] = (
    df["director"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .str.title()
)

df = df.drop_duplicates(subset=["movie_id"], keep="first").copy()

# 2.2 Standardize genres & fix typos
genre_map = {
    "scifi": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "acton": "Action",
    "action": "Action",
    "triller": "Thriller",
    "thriller": "Thriller",
    "drama": "Drama",
    "comedy": "Comedy",
    "horror": "Horror",
    "romance": "Romance",
    "animation": "Animation",
    "documentary": "Documentary",
    "fantasy": "Fantasy",
}

clean_genre_keys = (
    df["genre"]
    .fillna("unknown")
    .astype(str)
    .str.strip()
    .str.lower()
)
df["genre"] = clean_genre_keys.map(genre_map).fillna(clean_genre_keys.str.title())

# 2.3 Clean release_year (numeric conversion + 1900-2026 range check)
years_num = pd.to_numeric(df["release_year"].astype(str).str.strip(), errors="coerce")
df["release_year"] = pd.Series(
    np.where((years_num >= 1900) & (years_num <= 2026), np.round(years_num), np.nan),
    index=df.index,
).astype("Int64")

# 2.4 Clean rating (lowercase first, strip "/10" and "stars", enforce 0-10 bounds)
rating_str = (
    df["rating"]
    .astype(str)
    .str.lower()
    .str.replace("/10", "", regex=False)
    .str.replace("stars", "", regex=False)
    .str.replace("star", "", regex=False)
    .str.strip()
)
rating_num = pd.to_numeric(rating_str, errors="coerce")
df["rating"] = np.where((rating_num >= 0.0) & (rating_num <= 10.0), np.round(rating_num, 1), np.nan)

rows_before_rating_drop = len(df)
df = df.dropna(subset=["rating"]).copy()
dropped_invalid_ratings = rows_before_rating_drop - len(df)

# 2.5 Clean votes (handle commas, "k" suffix, negatives, and impute median)
votes_str = (
    df["votes"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(",", "", regex=False)
)
is_k = votes_str.str.endswith("k")
votes_num = pd.to_numeric(votes_str.str.rstrip("k").str.strip(), errors="coerce")
votes_scaled = np.where(is_k, votes_num * 1000, votes_num)
votes_valid = np.where(votes_scaled > 0, votes_scaled, np.nan)

votes_median = np.nanmedian(votes_valid)
df["votes"] = np.round(np.where(np.isnan(votes_valid), votes_median, votes_valid)).astype(int)

# 2.6 Clean duration_min (lowercase first, strip "min", enforce 30-300 bounds, impute median)
dur_str = (
    df["duration_min"]
    .astype(str)
    .str.lower()
    .str.replace("minutes", "", regex=False)
    .str.replace("min", "", regex=False)
    .str.strip()
)
dur_num = pd.to_numeric(dur_str, errors="coerce")
dur_valid = np.where((dur_num >= 30) & (dur_num <= 300), dur_num, np.nan)

dur_median = np.nanmedian(dur_valid)
df["duration_min"] = np.round(np.where(np.isnan(dur_valid), dur_median, dur_valid)).astype(int)

# 2.7 Clean budget_million (lowercase first, strip "$", "usd", "m", impute genre median)
budget_str = (
    df["budget_million"]
    .astype(str)
    .str.lower()
    .str.replace("$", "", regex=False)
    .str.replace("usd", "", regex=False)
    .str.replace("m", "", regex=False)
    .str.strip()
)
budget_num = pd.to_numeric(budget_str, errors="coerce")
df["budget_million"] = np.where(budget_num > 0, budget_num, np.nan)

genre_medians = df.groupby("genre")["budget_million"].transform("median")
overall_budget_median = np.nanmedian(df["budget_million"].to_numpy())
df["budget_million"] = (
    df["budget_million"]
    .fillna(genre_medians)
    .fillna(overall_budget_median)
    .round(1)
)

# Export cleaned CSV
df.to_csv("cleaned_movies_500.csv", index=False)

print("\n=== 2. CLEANING SUMMARY ===")
print(f"Original rows:           {len(df_raw)}")
print(f"Duplicates removed:      {len(df_raw) - rows_before_rating_drop}")
print(f"Invalid ratings dropped: {dropped_invalid_ratings}")
print(f"Final clean rows:        {len(df)}")
print("Exported cleaned data:   cleaned_movies_500.csv")


# ==========================================================
# 3. FEATURE ENGINEERING & RATING ANALYSIS
# ==========================================================
ratings_arr = df["rating"].to_numpy()
votes_arr = df["votes"].to_numpy()

# 3.1 IMDb-style Weighted Rating: WR = (v / (v + m)) * R + (m / (v + m)) * C
C = np.mean(ratings_arr)
m = np.percentile(votes_arr, 70)
df["weighted_rating"] = np.round(
    (votes_arr / (votes_arr + m)) * ratings_arr + (m / (votes_arr + m)) * C,
    2,
)

# 3.2 Rating Tiers using np.select
tier_conditions = [
    df["rating"] >= 8.0,
    df["rating"] >= 7.0,
    df["rating"] >= 6.0,
]
tier_labels = ["Excellent (8.0+)", "Good (7.0-7.9)", "Average (6.0-6.9)"]
df["rating_tier"] = np.select(tier_conditions, tier_labels, default="Below Average (<6.0)")

# 3.3 Z-score of ratings & Decade column
rating_std = np.std(ratings_arr, ddof=1)
df["rating_zscore"] = (
    np.round((ratings_arr - C) / rating_std, 2) if rating_std > 0 else 0.0
)
df["decade"] = ((df["release_year"] // 10) * 10).astype("Int64")


# ==========================================================
# 4. ANALYSIS REPORTS
# ==========================================================
q25, q50, q75 = np.percentile(ratings_arr, [25, 50, 75])

print("\n=== 3. OVERALL RATING STATISTICS ===")
print(f"Mean Rating:        {C:.2f}")
print(f"Median Rating:      {q50:.2f}")
print(f"Std Deviation:      {rating_std:.2f}")
print(f"25th / 75th Pct:    {q25:.2f} / {q75:.2f} (IQR: {q75 - q25:.2f})")
print(f"Min / Max Rating:   {np.min(ratings_arr):.1f} / {np.max(ratings_arr):.1f}")
print(f"70th Pct Votes (m): {int(m):,}")

print("\n=== 4. RATING TIER DISTRIBUTION ===")
tier_counts = df["rating_tier"].value_counts()
tier_pct = (df["rating_tier"].value_counts(normalize=True) * 100).round(1)
tier_table = pd.DataFrame({"count": tier_counts, "percent": tier_pct})
print(tier_table.to_string())

print("\n=== 5. TOP 10 MOVIES (BY WEIGHTED RATING) ===")
top_10 = df.sort_values(by="weighted_rating", ascending=False).head(10)
print(
    top_10[
        ["movie_id", "title", "genre", "release_year", "rating", "votes", "weighted_rating"]
    ].to_string(index=False)
)

print("\n=== 6. BOTTOM 5 LOWEST-RATED MOVIES ===")
bottom_5 = df.sort_values(by=["rating", "votes"], ascending=[True, False]).head(5)
print(
    bottom_5[
        ["movie_id", "title", "genre", "release_year", "rating", "votes"]
    ].to_string(index=False)
)

print("\n=== 7. GENRE PERFORMANCE SUMMARY ===")
genre_summary = (
    df.groupby("genre")
    .agg(
        movie_count=("movie_id", "count"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        median_votes=("votes", "median"),
        avg_budget_m=("budget_million", "mean"),
    )
    .round(2)
    .sort_values(by="avg_rating", ascending=False)
)
print(genre_summary.to_string())

print("\n=== 8. TOP DIRECTORS (MIN 15 MOVIES) ===")
director_summary = (
    df[df["director"] != "Unknown"]
    .groupby("director")
    .agg(
        movies=("movie_id", "count"),
        avg_rating=("rating", "mean"),
        avg_weighted=("weighted_rating", "mean"),
        total_votes=("votes", "sum"),
    )
    .query("movies >= 15")
    .round(2)
    .sort_values(by="avg_rating", ascending=False)
)
print(director_summary.to_string())

print("\n=== 9. DECADE TRENDS ===")
decade_summary = (
    df.dropna(subset=["decade"])
    .groupby("decade")
    .agg(
        movies=("movie_id", "count"),
        avg_rating=("rating", "mean"),
        avg_budget_m=("budget_million", "mean"),
        avg_duration=("duration_min", "mean"),
    )
    .round(2)
)
print(decade_summary.to_string())

print("\n=== 10. BUDGET VS. RATING (BUDGET QUARTILES) ===")
df["budget_tier"] = pd.qcut(
    df["budget_million"],
    q=4,
    labels=["Low Budget", "Mid-Low", "Mid-High", "Blockbuster"],
)
budget_analysis = (
    df.groupby("budget_tier", observed=False)
    .agg(
        movies=("movie_id", "count"),
        avg_budget_m=("budget_million", "mean"),
        avg_rating=("rating", "mean"),
        median_votes=("votes", "median"),
    )
    .round(2)
)
print(budget_analysis.to_string())

print("\n=== 11. CORRELATION MATRIX ===")
numeric_cols = ["rating", "votes", "duration_min", "budget_million", "release_year"]
corr_matrix = df[numeric_cols].corr().round(3)
print(corr_matrix.to_string())