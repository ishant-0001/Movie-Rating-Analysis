# Movie Rating Analysis

A command-line program that cleans a movie dataset and analyzes movie ratings using
Python, pandas, and NumPy.

## Problem

There are many movies to compare, and the raw data contains inconsistent values,
duplicates, and invalid ratings. This project cleans the dataset and summarizes
ratings, genres, directors, release decades, votes, and budgets.

## Questions answered

1. Which movies have the highest weighted ratings?
2. How are ratings distributed across rating tiers?
3. Which genres and directors have the highest ratings?
4. How do ratings, budgets, votes, durations, and release years relate?

## Dataset

The raw movie dataset is stored in `data/raw/`:

| File | Rows | Columns |
| --- | ---: | --- |
| `movies.csv` | 500 | 9 |

The columns are `movie_id`, `title`, `genre`, `director`, `release_year`,
`rating`, `votes`, `duration_min`, and `budget_million`.

## Method

1. **Inspect the data** - print the row and column counts, duplicate counts, and
   missing values.
2. **Clean the data**
   - normalize movie IDs, titles, directors, and genres
   - remove repeated movie IDs, keeping the first record
   - convert release years to numeric values and keep years from 1900 through 2026
   - normalize ratings, votes, durations, and budgets, treating invalid values as
     missing
   - remove movies with missing or out-of-range ratings
   - fill invalid or missing votes and durations with their medians, and fill
     missing budgets with the median for their genre (or the overall median)
3. **Save the cleaned data** to `data/processed/cleaned_movies_500.csv`. The raw
   file is not modified.
4. **Analyze ratings** - calculate a vote-weighted rating, rating tiers, rating
   z-scores, and release decades.
5. **Print reports** - display rating statistics, top and bottom movies, genre
   and director summaries, decade trends, budget quartiles, and correlations.

## Features

- Rank movies by an IMDb-style weighted rating that accounts for vote counts
- Show rating-tier distributions and overall rating statistics
- Summarize performance by genre, director, and release decade
- Compare budget quartiles and calculate a numeric correlation matrix
- Export the cleaned dataset for further analysis

## Project structure

```text
data/
├── raw/
│   └── movies.csv
└── processed/
    └── cleaned_movies_500.csv
code.py
project_code.py
requirements.txt
.gitignore
README.md
```

`project_code.py` is the project entry script. `code.py` contains the same
analysis for compatibility with the original project file.

## How to run

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the analysis from the project directory:

```bash
python project_code.py
```

The script reads the raw CSV and writes the cleaned CSV to the processed-data
directory, creating that directory if necessary.

## Sample result

The current dataset contains 500 input rows. The analysis retains 470 movies
after removing duplicate IDs and invalid ratings.

Top movies by weighted rating include:

| Rank | Movie | Rating | Votes | Weighted rating |
| ---: | --- | ---: | ---: | ---: |
| 1 | The Last Legacy | 8.7 | 51,200 | 8.19 |
| 2 | The Bright Orbit | 8.5 | 59,800 | 8.11 |
| 3 | The Lost Legacy | 8.6 | 45,120 | 8.09 |

## Key findings

- 25 repeated movie IDs were removed, and 5 movies with invalid ratings were
  dropped.
- The average rating among retained movies is 7.28, with a median of 7.30.
- `Good (7.0-7.9)` is the largest rating tier, with 227 movies.
- Animation has the highest average genre rating in this dataset (8.12).
- The highest weighted rating is 8.19, for *The Last Legacy*.

## Limitations

- Weighted ratings depend on the dataset-wide average rating and vote-count
  threshold, so rankings may change when the dataset changes.
- Missing votes, durations, and budgets are imputed; these values are estimates,
  not observed data.
- The analysis is descriptive and does not establish that budget, votes, or other
  factors cause a movie to receive a particular rating.

## Tools

Python 3 · pandas · NumPy
