


import pandas as pd
import numpy as np
import time
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION & SEED DATA
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(101)

GENRES = [
    "Drama", "Comedy", "Action", "Romance", "Thriller",
    "Horror", "Documentary", "Crime", "Sci-Fi", "Animation",
    "Fantasy", "Biography", "Mystery", "Family", "History"
]

COUNTRIES = [
    "United States", "India", "United Kingdom", "South Korea",
    "Spain", "France", "Japan", "Pakistan", "Turkey", "Canada",
    "Brazil", "Germany", "Mexico", "Italy", "Australia"
]

RATINGS = ["G", "PG", "PG-13", "TV-14", "TV-MA", "R", "TV-G", "TV-PG"]

DIRECTORS = [
    "Christopher Nolan", "Bong Joon-ho", "Greta Gerwig", "Ava DuVernay",
    "Steven Spielberg", "Chloe Zhao", "Jordan Peele", "Denis Villeneuve",
    "Ridley Scott", "Sofia Coppola", "James Cameron", "Patty Jenkins",
    "Martin Scorsese", "Kathryn Bigelow", "Wes Anderson", "Park Chan-wook",
    "Rian Johnson", "Taika Waititi", "Lulu Wang", "Mati Diop"
]

ACTORS = [
    "Meryl Streep", "Leonardo DiCaprio", "Viola Davis", "Tom Hanks",
    "Cate Blanchett", "Idris Elba", "Zendaya", "Ryan Reynolds",
    "Sandra Bullock", "Timothée Chalamet", "Anya Taylor-Joy", "Adam Driver",
    "Jennifer Lawrence", "Michael B. Jordan", "Florence Pugh", "Denzel Washington",
    "Margot Robbie", "Oscar Isaac", "Lupita Nyong'o", "Paul Mescal",
    "Ana de Armas", "Pedro Pascal", "Sydney Sweeney", "Austin Butler"
]

MOVIE_TITLES = [
    "The Last Horizon", "Echoes of Tomorrow", "Dark Waters",
    "Beneath the Surface", "The Quiet Storm", "Neon Dreams",
    "Shattered Glass", "The Forgotten Garden", "Between Two Worlds",
    "City of Shadows", "The Final Chapter", "Broken Promises",
    "Rising Tides", "The Invisible Thread", "Lost in Translation",
    "Heart of Darkness", "The Silver Lining", "Chasing Shadows",
    "Beyond the Stars", "The Long Way Home", "Paper Hearts",
    "Velvet Underground", "Midnight Run", "The Other Side",
    "Burning Bridges", "The Lucky Ones", "What We Leave Behind",
    "Parallel Lines", "The Last Dance", "Open Waters",
    "Into the Wild Blue", "Season of Change", "A Perfect Storm",
    "The Light Between", "Whispers in the Wind", "Golden Hour",
    "The Deep End", "Cold Comfort", "Uncharted Territory",
    "The Sum of All Parts", "Glass Houses", "Wild Hearts",
    "The Narrow Road", "Second Chances", "Out of the Blue",
    "The Painted Veil", "Electric Dreams", "Falling Forward",
    "The Weight of Words", "Almost Paradise", "Before Midnight",
    "Daughters of the Moon", "The Broken Circle", "New Horizons",
    "Crimson Peak", "The Distance Between", "One Last Summer",
    "Songs of Silence", "The Beautiful Ones", "Wasted Years",
    "Invisible Cities", "The Outer Limits", "Starfall",
    "Everything Changes", "The Ones We Love", "Under the Sun",
    "Spiral Staircase", "Bright Future", "The Empty Chair",
    "Mirrored Lives", "All That Glitters", "Fire and Water",
    "The Hidden Truth", "Surface Tension", "The Long Goodbye",
    "Beautiful Ruins", "Crossroads", "The Big Picture",
    "Somewhere Only We Know", "The Last Goodbye", "Blue Valentine"
]

SHOW_TITLES = [
    "Crown & Glory", "The Agency", "Dark Corner",
    "Love After Hours", "Rise of the Fallen", "The Witching Hour",
    "Code Red", "Scandal Files", "Parallel Worlds",
    "Bake My Day", "The Detective", "Her Story",
    "Wild District", "The Clinic", "Night Shift",
    "Outer Banks Remix", "The Diplomat", "Silk Road",
    "Blood & Honor", "Family First", "Startup Dreams",
    "The Inner Circle", "Signal Lost", "Drama Queen",
    "Cold Case Files", "The Fashion Week", "New Girl Energy"
]

DESCRIPTIONS = [
    "A gripping story about love, loss, and the choices that define us.",
    "When secrets unravel, no one is safe in this thrilling journey.",
    "Two strangers find themselves drawn together by fate and circumstance.",
    "An unlikely hero must rise against overwhelming odds to save everything.",
    "A family torn apart must confront the truth buried deep in their past.",
    "In a world where nothing is as it seems, one woman fights for justice.",
    "A young woman discovers her extraordinary gift — and its dangerous price.",
    "Friendship, betrayal, and redemption in a city that never sleeps.",
    "Based on true events that shocked the nation and changed history forever.",
    "A journey across continents in search of belonging, identity, and home.",
    "When a tech prodigy goes missing, her team races against time to find her.",
    "Three generations of women face the consequences of one summer's secret.",
    "A detective with a troubled past must solve the most personal case of her life.",
    "Ambition meets romance in the cutthroat world of high fashion.",
    "An immigrant family's story of struggle, love, and the American dream.",
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA GENERATION ENGINE
# Purpose: Build a realistic Netflix-style dataset using NumPy distributions
# ─────────────────────────────────────────────────────────────────────────────

def generate_netflix_dataset(n_movies=300, n_shows=150):
    """
    Generates a synthetic Netflix dataset with realistic distributions.

    Logic:
      - Movie durations: Normal distribution centred at 100 mins (realistic avg)
      - Release years: Skewed toward recent years (Netflix grew post-2015)
      - Genre combos: Real Netflix titles often span 2 genres
      - Ratings and countries sampled from weighted distributions

    Returns: pd.DataFrame — the master Netflix dataset
    """
    records = []

    # --- MOVIES ---
    for i in range(n_movies):
        title     = np.random.choice(MOVIE_TITLES) + (
                        f" {np.random.randint(2, 10)}" if i % 15 == 0 else "")
        director  = np.random.choice(DIRECTORS)
        cast_list = np.random.choice(ACTORS, size=np.random.randint(2, 5), replace=False)
        country   = np.random.choice(COUNTRIES,
                                     p=[0.25,0.15,0.1,0.08,0.07,0.07,
                                        0.05,0.04,0.04,0.04,
                                        0.03,0.03,0.02,0.02,0.01])
        year      = int(np.random.normal(2017, 4))
        year      = np.clip(year, 2000, 2023)
        duration  = int(np.random.normal(100, 20))
        duration  = np.clip(duration, 60, 210)

        # 2-genre combo for realism
        genre_pair = ", ".join(np.random.choice(GENRES, size=2, replace=False))
        rating     = np.random.choice(RATINGS,
                                      p=[0.05,0.1,0.2,0.2,0.2,0.15,0.05,0.05])
        description = np.random.choice(DESCRIPTIONS)
        score       = round(np.random.normal(6.8, 1.2), 1)
        score       = float(np.clip(score, 3.0, 10.0))

        month = np.random.choice(
            ["January","February","March","April","May","June",
             "July","August","September","October","November","December"])
        date_added = f"{month} {np.random.randint(1,28)}, {np.random.randint(2015,2024)}"

        records.append({
            "type":         "Movie",
            "title":        title,
            "director":     director,
            "cast":         ", ".join(cast_list),
            "country":      country,
            "date_added":   date_added,
            "release_year": year,
            "rating":       rating,
            "duration":     f"{duration} min",
            "listed_in":    genre_pair,
            "description":  description,
            "score":        score,
        })

    # --- TV SHOWS ---
    for j in range(n_shows):
        title     = np.random.choice(SHOW_TITLES) + (
                        f": Season {np.random.randint(1,5)}" if j % 8 == 0 else "")
        director  = np.random.choice(DIRECTORS)
        cast_list = np.random.choice(ACTORS, size=np.random.randint(3, 6), replace=False)
        country   = np.random.choice(COUNTRIES,
                                     p=[0.25,0.15,0.1,0.08,0.07,0.07,
                                        0.05,0.04,0.04,0.04,
                                        0.03,0.03,0.02,0.02,0.01])
        year      = int(np.random.normal(2018, 3))
        year      = np.clip(year, 2005, 2023)
        seasons   = int(np.random.exponential(2)) + 1
        seasons   = np.clip(seasons, 1, 10)

        genre_pair  = ", ".join(np.random.choice(GENRES, size=2, replace=False))
        rating      = np.random.choice(RATINGS,
                                       p=[0.05,0.1,0.2,0.2,0.2,0.15,0.05,0.05])
        description = np.random.choice(DESCRIPTIONS)
        score       = round(np.random.normal(7.0, 1.1), 1)
        score       = float(np.clip(score, 3.0, 10.0))

        month = np.random.choice(
            ["January","February","March","April","May","June",
             "July","August","September","October","November","December"])
        date_added = f"{month} {np.random.randint(1,28)}, {np.random.randint(2015,2024)}"

        records.append({
            "type":         "TV Show",
            "title":        title,
            "director":     director,
            "cast":         ", ".join(cast_list),
            "country":      country,
            "date_added":   date_added,
            "release_year": year,
            "rating":       rating,
            "duration":     f"{seasons} Season{'s' if seasons > 1 else ''}",
            "listed_in":    genre_pair,
            "description":  description,
            "score":        score,
        })

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA CLEANING MODULE
# Purpose: Handle missing/invalid data — core real-world Pandas skill
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(df):
    """
    Cleans the raw DataFrame:
      - Fills NaN values with sensible defaults
      - Strips whitespace from string columns
      - Extracts numeric duration for movies (e.g., "98 min" → 98)
      - Ensures release_year is integer
      - Drops exact duplicate rows

    This is the most critical real-world data science step —
    raw data is ALWAYS messy.

    Returns: cleaned df, movie-only df, show-only df
    """
    # Introduce 5% random NaN values to simulate real-world data
    for col in ["director", "cast", "country"]:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    # Fill missing values
    df["director"]  = df["director"].fillna("Unknown Director")
    df["cast"]      = df["cast"].fillna("Cast Unavailable")
    df["country"]   = df["country"].fillna("Unknown Country")

    # Strip whitespace
    df["title"]     = df["title"].str.strip()
    df["listed_in"] = df["listed_in"].str.strip()

    # Convert release_year to int safely
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000).astype(int)

    # Drop duplicates
    df = df.drop_duplicates(subset=["title", "type"])

    # Split into movie and show subsets
    movies = df[df["type"] == "Movie"].copy()
    shows  = df[df["type"] == "TV Show"].copy()

    # Extract numeric duration for movies
    movies["duration_mins"] = (
        movies["duration"]
        .str.replace(" min", "", regex=False)
        .str.strip()
        .apply(pd.to_numeric, errors="coerce")
        .fillna(90)
        .astype(int)
    )

    return df, movies, shows


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: ANALYTICS ENGINE
# Purpose: All statistical analysis using Pandas + NumPy
# ─────────────────────────────────────────────────────────────────────────────

def genre_analysis(df):
    """
    Finds the most common genres across the entire catalog.
    Uses Pandas Series.str.split() + explode() to handle comma-separated genres,
    then value_counts() for frequency ranking.
    """
    all_genres = (
        df["listed_in"]
        .str.split(", ")
        .explode()
        .str.strip()
        .value_counts()
        .reset_index()
    )
    all_genres.columns = ["Genre", "Count"]
    return all_genres.head(10)


def content_trend_by_year(df):
    """
    Counts how many titles Netflix added per release year.
    Uses Pandas groupby + size() — shows Netflix's content growth curve.
    """
    trend = (
        df.groupby(["release_year", "type"])
        .size()
        .reset_index(name="Count")
        .sort_values("release_year")
    )
    return trend


def country_production_stats(df):
    """
    Ranks countries by total content produced.
    Uses value_counts() on the country column.
    Also computes what % of the catalog each country contributes
    using NumPy division — demonstrates percentage calculation.
    """
    country_counts = df["country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Titles"]
    country_counts["Percentage"] = (
        np.round(country_counts["Titles"] / len(df) * 100, 1)
    )
    return country_counts.head(10)


def movie_duration_stats(movies):
    """
    Computes key statistics on movie durations using NumPy:
      - Mean, Median, Std deviation, Min, Max
      - Percentiles (25th, 75th) — shows distribution spread

    This demonstrates real NumPy statistical usage beyond just .mean()
    """
    durations = movies["duration_mins"].values  # Convert to NumPy array

    stats = {
        "Total Movies":       len(durations),
        "Average Duration":   f"{np.mean(durations):.1f} min",
        "Median Duration":    f"{np.median(durations):.1f} min",
        "Std Deviation":      f"{np.std(durations):.1f} min",
        "Shortest Movie":     f"{np.min(durations)} min",
        "Longest Movie":      f"{np.max(durations)} min",
        "25th Percentile":    f"{np.percentile(durations, 25):.0f} min",
        "75th Percentile":    f"{np.percentile(durations, 75):.0f} min",
    }
    return stats


def rating_distribution(df):
    """
    Breaks down content by rating category (e.g., PG-13, TV-MA).
    Uses value_counts() + percentage computation.
    """
    rating_counts = df["rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    rating_counts["Percentage"] = np.round(
        rating_counts["Count"] / len(df) * 100, 1
    )
    return rating_counts


def top_rated_content(df, content_type="Both", top_n=10):
    """
    Returns top N highest-scored content.
    Uses Pandas filtering + .nlargest() — efficient top-N selection.
    content_type: "Movie", "TV Show", or "Both"
    """
    if content_type == "Movie":
        subset = df[df["type"] == "Movie"]
    elif content_type == "TV Show":
        subset = df[df["type"] == "TV Show"]
    else:
        subset = df.copy()

    return (
        subset.nlargest(top_n, "score")
              [["title", "type", "listed_in", "release_year", "score", "country"]]
              .reset_index(drop=True)
    )


def actor_analysis(df, top_n=10):
    """
    Finds the most frequently appearing actors across all content.
    
    Logic:
      - Split the cast column (comma-separated) using str.split()
      - Use explode() to create one row per actor
      - Apply value_counts() to rank by appearances
    
    This is an advanced Pandas technique: handling list-like column data.
    """
    actor_counts = (
        df["cast"]
        .dropna()
        .str.split(", ")
        .explode()
        .str.strip()
        .value_counts()
        .reset_index()
    )
    actor_counts.columns = ["Actor", "Appearances"]
    return actor_counts[actor_counts["Actor"] != "Cast Unavailable"].head(top_n)


def director_analysis(df, top_n=10):
    """
    Ranks directors by number of titles in the catalog.
    Uses value_counts() on the director column.
    """
    director_counts = (
        df[df["director"] != "Unknown Director"]
        ["director"]
        .value_counts()
        .reset_index()
    )
    director_counts.columns = ["Director", "Titles"]
    return director_counts.head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RECOMMENDATION ENGINE
# Purpose: Genre-based and score-based content recommendation
# ─────────────────────────────────────────────────────────────────────────────

def recommend_by_genre(df, genre, content_type="Both", top_n=8):
    """
    Recommends content based on user's preferred genre.
    
    Logic:
      - Uses Pandas .str.contains() for flexible partial-match search
      - Filters by type if specified
      - Sorts by score descending so best content appears first
      - Returns top N results

    This is the core recommendation logic — genre + quality filtering.
    """
    if content_type == "Movie":
        subset = df[df["type"] == "Movie"]
    elif content_type == "TV Show":
        subset = df[df["type"] == "TV Show"]
    else:
        subset = df.copy()

    results = subset[
        subset["listed_in"].str.contains(genre, case=False, na=False)
    ].sort_values("score", ascending=False)

    return results[["title", "type", "listed_in", "release_year",
                     "score", "country", "description"]].head(top_n).reset_index(drop=True)


def recommend_by_country(df, country, top_n=8):
    """
    Recommends top-rated content from a specific country.
    Uses boolean masking + sort — simple but powerful Pandas pattern.
    """
    results = (
        df[df["country"].str.contains(country, case=False, na=False)]
        .sort_values("score", ascending=False)
        [["title", "type", "listed_in", "release_year", "score"]]
        .head(top_n)
        .reset_index(drop=True)
    )
    return results


def recommend_similar(df, title, top_n=6):
    """
    Finds content SIMILAR to a given title.
    
    Logic (manual cosine-like similarity without ML):
      1. Find the source title's genres
      2. Find all content that shares AT LEAST ONE genre
      3. Compute a Similarity Score = (matching genres / total genres) × score
      4. Sort by similarity score and return top N

    This demonstrates analytical thinking — a pseudo-recommendation system
    built purely with Pandas + NumPy, no ML library needed.
    """
    row = df[df["title"].str.lower() == title.lower()]
    if row.empty:
        # Try partial match
        row = df[df["title"].str.lower().str.contains(title.lower(), na=False)]
    if row.empty:
        return None, None

    source       = row.iloc[0]
    source_genres = set(str(source["listed_in"]).split(", "))

    def genre_similarity(genres_str):
        other_genres  = set(str(genres_str).split(", "))
        common        = len(source_genres & other_genres)
        total         = len(source_genres | other_genres)
        return common / total if total > 0 else 0

    candidate = df[df["title"] != source["title"]].copy()
    candidate["SimilarityScore"] = candidate["listed_in"].apply(genre_similarity)
    candidate["WeightedScore"]   = np.round(
        candidate["SimilarityScore"] * candidate["score"], 2
    )

    results = (
        candidate[candidate["SimilarityScore"] > 0]
        .sort_values("WeightedScore", ascending=False)
        [["title", "type", "listed_in", "score", "SimilarityScore"]]
        .head(top_n)
        .reset_index(drop=True)
    )
    return source, results


def search_by_actor(df, actor_name, top_n=8):
    """
    Search all content featuring a specific actor.
    Uses str.contains() — case-insensitive partial match.
    """
    results = (
        df[df["cast"].str.contains(actor_name, case=False, na=False)]
        .sort_values("score", ascending=False)
        [["title", "type", "listed_in", "release_year", "score", "cast"]]
        .head(top_n)
        .reset_index(drop=True)
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: WATCHLIST (FILE HANDLING)
# Purpose: Save/load user favorites to CSV — real-world persistence
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST_FILE = "my_watchlist.csv"

def load_watchlist():
    """Loads saved watchlist from CSV using Pandas read_csv."""
    if os.path.exists(WATCHLIST_FILE):
        return pd.read_csv(WATCHLIST_FILE)
    return pd.DataFrame(columns=["title", "type", "listed_in", "score", "added_on"])


def save_to_watchlist(df, title):
    """
    Adds a title to the personal watchlist CSV.
    Logic:
      - Search for title in dataset
      - If found, append to watchlist CSV (creates file if not exists)
      - Prevent duplicate entries using Pandas isin() check
    """
    row = df[df["title"].str.lower().str.contains(title.lower(), na=False)]
    if row.empty:
        return False, "Title not found in dataset."

    entry = row.iloc[0][["title", "type", "listed_in", "score"]].to_dict()
    entry["added_on"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    watchlist = load_watchlist()
    if entry["title"] in watchlist["title"].values:
        return False, f'"{entry["title"]}" is already in your watchlist!'

    new_row   = pd.DataFrame([entry])
    watchlist = pd.concat([watchlist, new_row], ignore_index=True)
    watchlist.to_csv(WATCHLIST_FILE, index=False)
    return True, entry["title"]


def remove_from_watchlist(title):
    """Removes a title from the watchlist CSV by title match."""
    watchlist = load_watchlist()
    if watchlist.empty:
        return False, "Watchlist is empty."
    before = len(watchlist)
    watchlist = watchlist[~watchlist["title"].str.lower().str.contains(title.lower(), na=False)]
    if len(watchlist) == before:
        return False, "Title not found in watchlist."
    watchlist.to_csv(WATCHLIST_FILE, index=False)
    return True, title


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DISPLAY / UI ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def divider(char="═", width=74):
    print(char * width)

def header(title):
    divider()
    pad = (72 - len(title)) // 2
    print("║" + " " * pad + title + " " * (72 - pad - len(title)) + "║")
    divider()

def loading(msg="Processing", steps=4):
    print(f"\n  ⏳ {msg}", end="", flush=True)
    for _ in range(steps):
        time.sleep(0.18)
        print(".", end="", flush=True)
    print(" Done!\n")

def print_df(df, title=None, max_col_width=30):
    if title:
        print(f"\n  📋 {title}")
        print("  " + "─" * 70)
    pd.set_option("display.max_colwidth", max_col_width)
    pd.set_option("display.width", 120)
    lines = df.to_string(index=True).split("\n")
    for line in lines:
        print("  " + line)
    print()

def print_stats(stats_dict, title=None):
    if title:
        print(f"\n  📊 {title}")
        print("  " + "─" * 50)
    for key, val in stats_dict.items():
        print(f"  {key:<25} : {val}")
    print()

def show_banner():
    clear()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ███╗   ██╗███████╗████████╗███████╗██╗     ██╗██╗  ██╗                 ║
║   ████╗  ██║██╔════╝╚══██╔══╝██╔════╝██║     ██║╚██╗██╔╝                 ║
║   ██╔██╗ ██║█████╗     ██║   █████╗  ██║     ██║ ╚███╔╝                  ║
║   ██║╚██╗██║██╔══╝     ██║   ██╔══╝  ██║     ██║ ██╔██╗                  ║
║   ██║ ╚████║███████╗   ██║   ██║     ███████╗██║██╔╝ ██╗                 ║
║   ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝     ╚══════╝╚═╝╚═╝  ╚═╝                 ║
║       BY : SADIA ILYAS                                                   ║
║       CONTENT  RECOMMENDATION  &  ANALYTICS  SYSTEM  v1.0                ║
║       ────────────────────────────────────────────────────               ║
║       Powered by Python  │  Pandas  │  NumPy                             ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MENU MODULES
# Each module is one interactive feature of the system
# ─────────────────────────────────────────────────────────────────────────────

def module_recommend_genre(df):
    """Module 1 — Recommend by Genre"""
    clear()
    header("GENRE-BASED RECOMMENDATION")
    print("\n  Popular Genres:")
    popular = ["Drama", "Comedy", "Action", "Romance", "Thriller",
               "Horror", "Crime", "Sci-Fi", "Documentary", "Animation"]
    for i, g in enumerate(popular, 1):
        print(f"    [{i:2}] {g}")

    genre = input("\n  Enter genre (or type your own): ").strip()
    if not genre:
        print("  ⚠ No genre entered.")
        return

    print("\n  Filter by type?")
    print("    [1] Movies only  [2] TV Shows only  [3] Both")
    t = input("  Choice: ").strip()
    ctype = {"1": "Movie", "2": "TV Show"}.get(t, "Both")

    loading(f"Finding best {genre} content")
    results = recommend_by_genre(df, genre, ctype)

    if results.empty:
        print(f"\n  ❌ No content found for genre: '{genre}'")
    else:
        print_df(results[["title", "type", "score", "release_year", "country"]],
                 f"Top Recommendations — {genre}")


def module_recommend_country(df):
    """Module 2 — Recommend by Country"""
    clear()
    header("COUNTRY-BASED RECOMMENDATION")
    print("\n  Top Content-Producing Countries:")
    for i, c in enumerate(COUNTRIES[:10], 1):
        print(f"    [{i:2}] {c}")

    country = input("\n  Enter country name: ").strip()
    if not country:
        return

    loading(f"Searching top content from {country}")
    results = recommend_by_country(df, country)

    if results.empty:
        print(f"\n  ❌ No content found from '{country}'")
    else:
        print_df(results, f"Top-Rated Content from {country}")


def module_similar_titles(df):
    """Module 3 — Find Similar Titles"""
    clear()
    header("SIMILAR CONTENT FINDER")
    print("\n  Enter a title you enjoyed and we'll find similar content.\n")

    # Show a few sample titles
    sample = df["title"].sample(5, random_state=1).tolist()
    print("  Sample titles in dataset:")
    for t in sample:
        print(f"    • {t}")

    title = input("\n  Enter title: ").strip()
    if not title:
        return

    loading("Analyzing content similarity")
    source, results = recommend_similar(df, title)

    if source is None:
        print(f"\n  ❌ Title '{title}' not found. Try a different name.")
        return

    print(f"\n  🎬 Based on: {source['title']}")
    print(f"     Genre   : {source['listed_in']}")
    print(f"     Score   : {source['score']}")
    print()
    print_df(results, "You Might Also Like")


def module_top_rated(df):
    """Module 4 — Top Rated Content"""
    clear()
    header("TOP RATED CONTENT")
    print("\n  View top rated:")
    print("    [1] Movies   [2] TV Shows   [3] All")
    choice = input("  Choice: ").strip()
    ctype  = {"1": "Movie", "2": "TV Show"}.get(choice, "Both")

    loading("Ranking by score")
    results = top_rated_content(df, ctype, top_n=10)
    print_df(results, f"Top 10 Highest Rated — {ctype}")


def module_genre_analysis(df):
    """Module 5 — Genre Trend Analysis"""
    clear()
    header("GENRE ANALYSIS")
    loading("Counting genre frequencies")
    g = genre_analysis(df)
    print_df(g, "Top 10 Most Common Genres in Catalog")

    # Movie vs TV Show ratio using NumPy
    movie_count = len(df[df["type"] == "Movie"])
    show_count  = len(df[df["type"] == "TV Show"])
    ratio       = np.round(movie_count / show_count, 2)

    print(f"  🎬 Movies   : {movie_count}")
    print(f"  📺 TV Shows : {show_count}")
    print(f"  📐 Ratio    : {ratio} movies per TV show\n")


def module_duration_stats(movies):
    """Module 6 — Movie Duration Statistics"""
    clear()
    header("MOVIE DURATION ANALYSIS")
    loading("Computing duration statistics")
    stats = movie_duration_stats(movies)
    print_stats(stats, "Movie Duration Statistics (NumPy)")

    # Bucket movies into duration ranges using np.digitize
    bins    = [0, 80, 100, 120, 150, 999]
    labels  = ["< 80 min", "80–100 min", "100–120 min", "120–150 min", "150+ min"]
    durations = movies["duration_mins"].values
    bucketed  = np.digitize(durations, bins) - 1
    bucketed  = np.clip(bucketed, 0, len(labels) - 1)

    print("  📦 Duration Distribution:")
    print("  " + "─" * 40)
    unique, counts = np.unique(bucketed, return_counts=True)
    for idx, count in zip(unique, counts):
        bar = "█" * (count // 3)
        print(f"  {labels[idx]:<14} : {count:3d}  {bar}")
    print()


def module_country_stats(df):
    """Module 7 — Country Production Analysis"""
    clear()
    header("COUNTRY PRODUCTION STATS")
    loading("Analyzing production by country")
    c = country_production_stats(df)
    print_df(c, "Top 10 Content-Producing Countries")


def module_actor_search(df):
    """Module 8 — Actor Search"""
    clear()
    header("ACTOR / CAST SEARCH")

    print("\n  [1] Search by actor name")
    print("  [2] See most appearing actors")
    choice = input("\n  Choice: ").strip()

    if choice == "1":
        actor = input("  Enter actor name: ").strip()
        if not actor:
            return
        loading(f"Searching titles featuring {actor}")
        results = search_by_actor(df, actor)
        if results.empty:
            print(f"\n  ❌ No content found featuring '{actor}'")
        else:
            print_df(results[["title", "type", "score", "listed_in"]],
                     f"Titles featuring {actor}")

    elif choice == "2":
        loading("Counting actor appearances")
        top_actors = actor_analysis(df, top_n=12)
        print_df(top_actors, "Most Appearing Actors in Catalog")


def module_year_trend(df):
    """Module 9 — Content by Year"""
    clear()
    header("CONTENT RELEASE TREND BY YEAR")
    loading("Analyzing year-wise content growth")
    trend = content_trend_by_year(df)

    # Use NumPy to find the peak year
    movie_trend = trend[trend["type"] == "Movie"]
    show_trend  = trend[trend["type"] == "TV Show"]

    peak_movie_year = int(movie_trend.loc[movie_trend["Count"].idxmax(), "release_year"])
    peak_show_year  = int(show_trend.loc[show_trend["Count"].idxmax(), "release_year"])

    print(f"  🎬 Peak Movie Release Year : {peak_movie_year}")
    print(f"  📺 Peak TV Show Year       : {peak_show_year}\n")

    # Bar chart in console
    year_total = trend.groupby("release_year")["Count"].sum().reset_index()
    year_total = year_total[year_total["release_year"] >= 2010]

    print("  📊 Content Added Per Year (2010 onwards):")
    print("  " + "─" * 55)
    max_count = year_total["Count"].max()
    for _, row in year_total.iterrows():
        bar_len = int((row["Count"] / max_count) * 35)
        bar     = "█" * bar_len
        print(f"  {int(row['release_year'])} │ {bar:<35} {int(row['Count'])}")
    print()


def module_rating_breakdown(df):
    """Module 10 — Rating Breakdown"""
    clear()
    header("CONTENT RATING BREAKDOWN")
    loading("Computing rating distribution")
    r = rating_distribution(df)
    print_df(r, "Content by Rating Category")

    print("  💡 Rating Guide:")
    print("  G = All ages | PG = Parental Guidance | PG-13 = 13+")
    print("  R = Restricted | TV-MA = Mature Audiences\n")


def module_watchlist(df):
    """Module 11 — My Watchlist"""
    clear()
    header("MY PERSONAL WATCHLIST")
    print("\n  [1] View watchlist")
    print("  [2] Add title to watchlist")
    print("  [3] Remove title from watchlist")
    choice = input("\n  Choice: ").strip()

    if choice == "1":
        wl = load_watchlist()
        if wl.empty:
            print("\n  📭 Your watchlist is empty. Start adding titles!\n")
        else:
            print_df(wl, f"Your Watchlist ({len(wl)} titles)")

    elif choice == "2":
        title = input("  Enter title to add: ").strip()
        if title:
            success, msg = save_to_watchlist(df, title)
            if success:
                print(f'\n  ✅ Added "{msg}" to your watchlist!\n')
            else:
                print(f"\n  ⚠ {msg}\n")

    elif choice == "3":
        title = input("  Enter title to remove: ").strip()
        if title:
            success, msg = remove_from_watchlist(title)
            if success:
                print(f'\n  🗑 Removed "{msg}" from watchlist.\n')
            else:
                print(f"\n  ⚠ {msg}\n")


def module_summary(df, movies, shows):
    """Module 12 — Full Dataset Summary"""
    clear()
    header("COMPLETE DATASET SUMMARY")
    loading("Compiling full analytics report")

    stats = {
        "Total Titles":          len(df),
        "Movies":                len(movies),
        "TV Shows":              len(shows),
        "Countries Represented": df["country"].nunique(),
        "Unique Directors":      df[df["director"] != "Unknown Director"]["director"].nunique(),
        "Year Range":            f"{df['release_year'].min()} – {df['release_year'].max()}",
        "Avg Movie Duration":    f"{movies['duration_mins'].mean():.0f} min",
        "Highest Rated Title":   df.loc[df['score'].idxmax(), 'title'],
        "Highest Score":         df['score'].max(),
        "Avg Content Score":     f"{df['score'].mean():.2f} / 10",
        "Most Common Genre":     df['listed_in'].str.split(', ').explode().value_counts().idxmax(),
        "Top Country":           df['country'].value_counts().idxmax(),
    }

    print_stats(stats, "Netflix Catalog Overview")

    print("  📈 Score Distribution (NumPy Histogram):")
    print("  " + "─" * 50)
    scores   = df["score"].values
    hist, edges = np.histogram(scores, bins=7, range=(3, 10))
    for count, edge in zip(hist, edges):
        bar = "█" * (count // 4)
        print(f"  Score {edge:.1f}+ │ {bar:<20} {count}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MAIN MENU & APPLICATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    show_banner()
    loading("Generating Netflix catalog dataset", steps=5)
    loading("Cleaning and preprocessing data",   steps=4)
    loading("Building recommendation index",     steps=3)

    df, movies, shows = clean_dataset(generate_netflix_dataset(300, 150))

   
    MENU = {
        "1":  ("🎬 Recommend by Genre",           lambda: module_recommend_genre(df)),
        "2":  ("🌍 Recommend by Country",         lambda: module_recommend_country(df)),
        "3":  ("🔎 Find Similar Titles",          lambda: module_similar_titles(df)),
        "4":  ("⭐ Top Rated Content",            lambda: module_top_rated(df)),
        "5":  ("📊 Genre Trend Analysis",         lambda: module_genre_analysis(df)),
        "6":  ("⏱ Movie Duration Statistics",    lambda: module_duration_stats(movies)),
        "7":  ("🗺 Country Production Stats",     lambda: module_country_stats(df)),
        "8":  ("🎭 Actor / Cast Search",          lambda: module_actor_search(df)),
        "9":  ("📅 Content by Release Year",      lambda: module_year_trend(df)),
        "10": ("🔞 Rating Breakdown",             lambda: module_rating_breakdown(df)),
        "11": ("❤  My Watchlist",                lambda: module_watchlist(df)),
        "12": ("📁 Full Dataset Summary",         lambda: module_summary(df, movies, shows)),
        "0":  ("Exit",                            None),
    }

    while True:
        print("\n" + "═" * 74)
        print("  MAIN MENU — Netflix Content Recommendation & Analytics System")
        print("═" * 74)
        for key, (label, _) in MENU.items():
            print(f"    [{key:>2}]  {label}")
        print("═" * 74)

        choice = input("\n  Enter your choice: ").strip()

        if choice == "0":
            print("\n  👋 Thanks for using Netflix Analyzer. Happy watching!\n")
            break
        elif choice in MENU and MENU[choice][1]:
            MENU[choice][1]()
            input("\n  Press ENTER to return to main menu...")
        else:
            print("\n  ⚠ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
