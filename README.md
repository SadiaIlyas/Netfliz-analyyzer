# Netflix Recommendation & Analytics System 🎬

> A fully interactive Netflix-style recommendation and analytics platform built using Python, Pandas, and NumPy — featuring synthetic dataset generation, recommendation engine, statistical analysis, watchlist management, and advanced content analytics.

<p>
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Data_Analysis-Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Numerical_Computing-NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/System-CLI_Based-2E8B57?style=flat-square"/>
  <img src="https://img.shields.io/badge/Recommendation-Engine-FF6B35?style=flat-square"/>
  <img src="https://img.shields.io/badge/Analytics-Advanced-5C2D91?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Complete-2E8B57?style=flat-square"/>
</p>

**GC University Lahore — Department of Computer Science**  
Submitted by: **Sadia Ilyas** (0078-BSCS-24)  
Project Type: Python Data Analytics & Recommendation System

---

## Project Overview

This project simulates a real-world Netflix-style recommendation and analytics platform using Python. The system generates a fully synthetic streaming-content dataset and performs advanced analysis using Pandas and NumPy.

The platform integrates multiple modules including:
- Genre-based recommendation engine
- Similar-content recommendation system
- Country and actor-based search
- Statistical analytics dashboard
- Watchlist management system
- Trend analysis and rating breakdown
- Dataset summarization and insights

The project demonstrates practical data science workflows:
- dataset generation
- data cleaning
- preprocessing
- recommendation logic
- statistical computation
- analytics reporting
- CSV-based persistence

---

## System Modules

| Module | Features | Responsibility |
|---|---|---|
| **Dataset Generation Engine** | Synthetic Netflix catalog generation | Creates realistic movies & TV shows using NumPy distributions |
| **Data Cleaning Module** | NaN handling, preprocessing, duplicate removal | Cleans and structures raw dataset |
| **Recommendation Engine** | Genre recommendation, country recommendation, similar content | Personalized content suggestions |
| **Analytics Engine** | Genre trends, rating analysis, actor analysis, duration statistics | Statistical insights and reporting |
| **Search Engine** | Actor search, title matching, filtering | Flexible content retrieval |
| **Watchlist System** | Add/remove/view watchlist | CSV-based user persistence |
| **CLI User Interface** | Interactive menu-driven system | User interaction and navigation |
| **Summary Dashboard** | Full catalog statistics | Overall dataset insights |

---

## Core Features

### 🎬 Recommendation Features
- Genre-based recommendation
- Country-based recommendation
- Similar title recommendation
- Top-rated content finder
- Actor/cast-based search

### 📊 Analytics Features
- Genre trend analysis
- Content release trend by year
- Country production analysis
- Rating distribution analysis
- Movie duration statistics
- Actor appearance ranking
- Director ranking system

### ❤ User Features
- Personal watchlist
- Add/remove favorite titles
- Persistent CSV storage
- Interactive command-line interface

### ⚙ Data Science Features
- Synthetic dataset generation
- Data cleaning and preprocessing
- Statistical analysis using NumPy
- Advanced Pandas transformations
- Similarity scoring system

---

## Dataset Design Methodology

### Step 1 — Synthetic Data Generation
The dataset was generated using NumPy random distributions to simulate realistic Netflix-style content.

Examples:
- Movie durations follow normal distribution centered around 100 minutes
- Release years skew toward recent years (post-2015 growth)
- TV show seasons generated using exponential distribution
- Ratings and countries sampled using weighted probabilities

### Step 2 — Realistic Content Modeling
Each title includes:
- title
- director
- cast
- country
- release year
- rating
- duration
- genres
- description
- content score

Genre combinations were intentionally designed as multi-label categories to simulate real streaming platforms.

### Step 3 — Data Cleaning & Preprocessing
The system introduces random missing values to mimic real-world messy datasets.

Cleaning operations include:
- Filling NaN values
- Removing duplicates
- String normalization
- Numeric conversion
- Feature extraction

### Step 4 — Recommendation Logic
A similarity-based recommendation engine was implemented using:
- genre overlap scoring
- weighted content scores
- filtering and ranking

The system demonstrates recommendation-system logic without external machine learning libraries.

---

## Dataset Statistics

| Metric | Count |
|---|---|
| Total Titles | 450+ |
| Movies | 300 |
| TV Shows | 150 |
| Genres | 15 |
| Countries | 15 |
| Directors | 20 |
| Actors | 24 |
| Recommendation Modules | 5 |
| Analytics Modules | 7 |
| Watchlist Features | 3 |

---

## Key Python & Pandas Highlights

### Genre Frequency Analysis
```python
all_genres = (
    df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
)
Top Rated Content Selection
top_content = (
    subset.nlargest(top_n, "score")
)
Similarity Recommendation Engine
candidate["SimilarityScore"] = (
    candidate["listed_in"].apply(genre_similarity)
)
NumPy Statistical Analysis
stats = {
    "Average Duration": np.mean(durations),
    "Median Duration": np.median(durations),
    "Std Deviation": np.std(durations),
}
Watchlist CSV Persistence
watchlist.to_csv(WATCHLIST_FILE, index=False)
Recommendation System Logic
Genre-Based Recommendation

The engine filters content matching a preferred genre and ranks results by score.

Similar Content Recommendation

Similarity is calculated using:

Similarity Score =
Matching Genres / Total Genres

Final ranking:

Weighted Score =
Similarity Score × Content Rating

This creates a lightweight recommendation engine without machine learning frameworks.

Analytics Demonstrated
Analysis Type	Technique Used
Genre Frequency	explode() + value_counts()
Country Statistics	value_counts() + percentages
Duration Analysis	NumPy statistics
Year Trends	groupby() + aggregation
Rating Breakdown	distribution analysis
Similarity Analysis	custom scoring system
Histogram Analysis	NumPy histogram
Repository Structure
Netflix-Recommendation-Analytics-System/
│
├── data/
│   └── my_watchlist.csv
│
├── screenshots/
│   ├── main_menu.png
│   ├── recommendations.png
│   ├── analytics.png
│   └── watchlist.png
│
├── netflix_analyzer.py
├── requirements.txt
├── README.md
└── LICENSE
Installation
Clone Repository
git clone https://github.com/SadiaIlyas/netflix-recommendation-analytics-system.git
Install Requirements
pip install -r requirements.txt
Run Project
python netflix_analyzer.py
Technologies Used
Technology	Purpose
Python	Core programming language
Pandas	Data analysis and manipulation
NumPy	Statistical and numerical analysis
CSV	Persistent watchlist storage
OS / Time / Warnings	System utilities and UI
Concepts Demonstrated
Data Generation — realistic synthetic dataset generation using NumPy distributions
Data Cleaning — handling missing values, duplicates, and inconsistent formatting
Data Analysis — filtering, grouping, aggregation, and trend analysis using Pandas
Recommendation Systems — genre similarity scoring and weighted ranking
Statistical Analysis — mean, median, percentiles, histogram analysis
File Handling — CSV persistence for user watchlists
CLI Development — interactive menu-driven application
Data Modeling — structured content representation similar to streaming platforms
Advanced Pandas Techniques Used
groupby()
explode()
value_counts()
nlargest()
str.contains()
boolean masking
aggregation
sorting
DataFrame concatenation
Advanced NumPy Techniques Used
np.mean()
np.median()
np.std()
np.percentile()
np.histogram()
np.clip()
np.random.normal()
np.random.exponential()
Future Improvements
Planned Upgrades
GUI using Tkinter
Web dashboard using Streamlit
Real Netflix dataset integration
Machine learning recommendation engine
TF-IDF content similarity
Cosine similarity recommendations
Graph visualization using Matplotlib
User authentication system
AI Upgrade Path
Current Recommendation System
            ↓
TF-IDF Vectorization
            ↓
Cosine Similarity
            ↓
AI Recommendation Engine
What I Learned
How real-world datasets are generated, cleaned, and analyzed
Why preprocessing is the most important part of data science
That recommendation systems can be built even without machine learning libraries
How Pandas transforms raw data into meaningful insights
Why NumPy is essential for fast statistical computation
How file persistence works using CSV storage
How to structure a large Python project into modules and analytics pipelines
That data analysis is not just graphs — it is understanding patterns, distributions, and relationships
Project Highlights

✔ Fully interactive CLI system
✔ Synthetic Netflix-style dataset generation
✔ Recommendation engine implementation
✔ Statistical analysis using NumPy
✔ Advanced Pandas operations
✔ CSV-based persistent watchlist
✔ Real-world data cleaning workflow
✔ Modular analytics architecture

Authors

Sadia Ilyas — 0078-BSCS-24 | CS Student @ GCU Lahore
LinkedIn · GitHub
