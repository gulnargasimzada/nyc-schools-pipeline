"""
NYC Schools pipeline.

Ingests two related NYC Open Data datasets (math test results and school
demographics), cleans and merges them, loads the result into PostgreSQL,
and analyzes the processed data.
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import TaskGroup, dag, task

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# Shared folder mounted into every Airflow container (see docker-compose.yaml)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))
RAW_DIR = DATA_DIR / "raw"               # Untouched downloads
STAGING_DIR = DATA_DIR / "staging"       # Intermediate cleaned/transformed files
PROCESSED_DIR = DATA_DIR / "processed"   # Final merged dataset
ANALYSIS_DIR = DATA_DIR / "analysis"     # Charts and model results

# Source datasets from NYC Open Data (both identify schools by DBN)
SOURCES = {
    "math": "https://data.cityofnewyork.us/api/views/74kb-55u9/rows.csv?accessType=DOWNLOAD",
    "demographics": "https://data.cityofnewyork.us/api/views/c7ru-d68s/rows.csv?accessType=DOWNLOAD",
}

# Demographic columns to keep: original name -> standardized name
DEMO_COLUMNS = {
    "DBN": "dbn",
    "School Name": "school_name",
    "Total Enrollment": "total_enrollment",
    "% Female": "pct_female",
    "% Students with Disabilities": "pct_students_with_disabilities",
    "% English Language Learners": "pct_ell",
    "% Poverty": "pct_poverty",
    "Economic Need Index": "economic_need_index",
    "% Asian": "pct_asian",
    "% Black": "pct_black",
    "% Hispanic": "pct_hispanic",
    "% White": "pct_white",
    "% Multi-Racial": "pct_multi_racial",
}

# Columns stored as text percentages such as "84.7%" or "Above 95%"
TEXT_PERCENT_COLUMNS = ["% Poverty", "Economic Need Index"]

# Warehouse table in the schools_db database (Airflow Connection: schools_db)
TABLE_NAME = "school_math_demographics"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    dbn                            TEXT PRIMARY KEY,
    school_year                    TEXT NOT NULL,
    school_name                    TEXT,
    number_tested                  INTEGER,
    mean_scale_score               DOUBLE PRECISION,
    pct_proficient                 DOUBLE PRECISION NOT NULL,
    total_enrollment               INTEGER,
    pct_female                     DOUBLE PRECISION,
    pct_students_with_disabilities DOUBLE PRECISION,
    pct_ell                        DOUBLE PRECISION,
    pct_poverty                    DOUBLE PRECISION,
    economic_need_index            DOUBLE PRECISION,
    pct_asian                      DOUBLE PRECISION,
    pct_black                      DOUBLE PRECISION,
    pct_hispanic                   DOUBLE PRECISION,
    pct_white                      DOUBLE PRECISION,
    pct_multi_racial               DOUBLE PRECISION,
    loaded_at                      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# The 3rd character of a DBN identifies the borough (e.g. 01M015 -> M -> Manhattan)
BOROUGHS = {"M": "Manhattan", "X": "Bronx", "K": "Brooklyn", "Q": "Queens", "R": "Staten Island"}

# Features for the baseline model. Excluded on purpose:
#   - pct_poverty: nearly duplicates economic_need_index
#   - race/ethnicity shares: mostly proxy for economic need; fairness concerns
#   - mean_scale_score, number_tested: come from the test itself (data leakage)
MODEL_FEATURES = [
    "economic_need_index",
    "pct_ell",
    "pct_students_with_disabilities",
    "pct_female",
    "total_enrollment",
]

# Applied to every task: retry temporary failures such as HTTP 503
default_args = {
    "owner": "data-engineering",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}


def read_table(table_name: str):
    """Read a whole warehouse table from PostgreSQL into a pandas DataFrame."""
    import pandas as pd
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    hook = PostgresHook(postgres_conn_id="schools_db")
    conn = hook.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name};")
            columns = [col[0] for col in cur.description]  # Column names of the result
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


@dag(
    dag_id="nyc_schools_pipeline",
    schedule="@daily",                    # Run once a day at midnight (UTC)
    start_date=datetime(2026, 11, 1),
    catchup=False,                        # Do not backfill missed past runs
    max_active_runs=1,                    # Never run two copies at the same time
    default_args=default_args,
    tags=["nyc-schools", "assignment"],
)
def nyc_schools_pipeline():

    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------
    @task
    def extract(dataset_name: str) -> str:
        """Download one source CSV to data/raw/ and return its local path."""
        import requests  # Imported inside the task to keep DAG parsing fast

        url = SOURCES[dataset_name]
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        target = RAW_DIR / f"{dataset_name}.csv"
        temp_file = RAW_DIR / f"{dataset_name}.csv.part"

        # Stream the download in 1 MB chunks to a temporary file
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()  # Fail the task on 4xx/5xx (Airflow will retry)
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

        if temp_file.stat().st_size == 0:
            raise ValueError(f"Downloaded file for '{dataset_name}' is empty")

        # Replace the old file only after the download fully succeeded
        temp_file.replace(target)

        size_mb = target.stat().st_size / 1e6
        print(f"Saved '{dataset_name}': {size_mb:.1f} MB -> {target}")
        return str(target)

    # -----------------------------------------------------------------
    # Math dataset: clean and transform
    # -----------------------------------------------------------------
    @task
    def clean_math(raw_path: str) -> str:
        """Keep one row per school for 2021-22 and fix data types."""
        import pandas as pd

        df = pd.read_csv(raw_path, dtype=str)
        print(f"Raw rows: {len(df):,}")

        # 1. Keep only school-level rows for the 2021-22 school year (2022 exam).
        #    Charter schools are not reported at this level for 2022, so only "School" is kept.
        df = df[
            (df["Report Category"] == "School")
            & (df["Year"] == "2022")
            & (df["Grade"] == "All Grades")
            & (df["Student Category"] == "All Students")
        ].copy()
        print(f"After filtering (schools, 2022, All Grades, All Students): {len(df):,}")

        # 2. Clean the school ID and convert numeric columns.
        #    Suppressed values ('s') cannot be converted and become NaN.
        df["Geographic Subdivision"] = df["Geographic Subdivision"].str.strip()
        for col in ["Number Tested", "Mean Scale Score", "Pct Level 3 and 4"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. Drop schools without a proficiency rate (it is the basis of the ML target)
        missing = df["Pct Level 3 and 4"].isna().sum()
        df = df.dropna(subset=["Pct Level 3 and 4"])
        print(f"Dropped {missing} schools with suppressed results; remaining: {len(df):,}")

        # 4. Data quality check: each school must appear exactly once
        duplicates = df["Geographic Subdivision"].duplicated().sum()
        if duplicates:
            raise ValueError(f"Found {duplicates} duplicated school IDs in math data")

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        out_path = STAGING_DIR / "math_clean.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    @task
    def transform_math(clean_path: str) -> str:
        """Rename columns, convert units and keep only the columns we need."""
        import pandas as pd

        df = pd.read_csv(clean_path, dtype={"Geographic Subdivision": str})

        # Standardize column names (snake_case, shared key name "dbn")
        df = df.rename(columns={
            "Geographic Subdivision": "dbn",
            "Number Tested": "number_tested",
            "Mean Scale Score": "mean_scale_score",
            "Pct Level 3 and 4": "pct_proficient",
        })

        # Convert percentage (0-100) to a fraction (0-1) to match the demographics data
        df["pct_proficient"] = df["pct_proficient"] / 100
        df["number_tested"] = df["number_tested"].astype(int)

        # Keep only the columns needed downstream
        df = df[["dbn", "number_tested", "mean_scale_score", "pct_proficient"]]

        # Data quality check: proficiency must be a valid fraction
        if not df["pct_proficient"].between(0, 1).all():
            raise ValueError("pct_proficient contains values outside the 0-1 range")

        print(f"Transformed math data: {len(df):,} schools")
        print(df.head())

        out_path = STAGING_DIR / "math_transformed.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    # -----------------------------------------------------------------
    # Demographics dataset: clean and transform
    # -----------------------------------------------------------------
    @task
    def clean_demographics(raw_path: str) -> str:
        """Keep one row per school for 2021-22 and fix data types."""
        import pandas as pd

        df = pd.read_csv(raw_path, dtype=str)
        print(f"Raw rows: {len(df):,}")

        # 1. Keep only the 2021-22 school year (matches the 2022 math exam)
        df = df[df["Year"] == "2021-22"].copy()
        print(f"After filtering to 2021-22: {len(df):,}")

        # 2. Clean the school ID
        df["DBN"] = df["DBN"].str.strip()

        # 3. Text percentages: "Above 95%" is a censored value, so we replace it
        #    with its lower bound (95%). Then remove the % sign: "84.7%" -> "84.7"
        for col in TEXT_PERCENT_COLUMNS:
            capped = (df[col] == "Above 95%").sum()
            df[col] = (
                df[col]
                .str.replace("Above 95%", "95%", regex=False)
                .str.replace("%", "", regex=False)
            )
            print(f"'{col}': replaced {capped} 'Above 95%' values with 95")

        # 4. Convert every kept numeric column from text to numbers
        numeric_columns = [c for c in DEMO_COLUMNS if c not in ("DBN", "School Name")]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        missing = df[numeric_columns].isna().sum()
        print("Missing values after conversion:", missing[missing > 0].to_dict() or "none")

        # 5. Data quality check: each school must appear exactly once
        duplicates = df["DBN"].duplicated().sum()
        if duplicates:
            raise ValueError(f"Found {duplicates} duplicated DBN values in demographics data")

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        out_path = STAGING_DIR / "demographics_clean.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    @task
    def transform_demographics(clean_path: str) -> str:
        """Keep the needed columns, rename them and put all percentages on a 0-1 scale."""
        import pandas as pd

        df = pd.read_csv(clean_path, dtype={"DBN": str})

        # Keep only the selected columns and give them standardized names
        df = df[list(DEMO_COLUMNS)].rename(columns=DEMO_COLUMNS)

        # These two were on a 0-100 scale; all other percentages are already 0-1
        df["pct_poverty"] = df["pct_poverty"] / 100
        df["economic_need_index"] = df["economic_need_index"] / 100

        # Data quality check: every percentage must be a valid fraction (NaN allowed)
        fraction_columns = [c for c in df.columns if c.startswith("pct_")] + ["economic_need_index"]
        for col in fraction_columns:
            if not df[col].dropna().between(0, 1).all():
                raise ValueError(f"{col} contains values outside the 0-1 range")

        print(f"Transformed demographics data: {len(df):,} schools, {len(df.columns)} columns")
        print(df.head())

        out_path = STAGING_DIR / "demographics_transformed.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    # -----------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------
    @task
    def merge_datasets(math_path: str, demographics_path: str) -> str:
        """Join math results and demographics on the school ID (DBN)."""
        import pandas as pd

        math = pd.read_csv(math_path, dtype={"dbn": str})
        demo = pd.read_csv(demographics_path, dtype={"dbn": str})

        # Report how well the two datasets overlap before joining
        math_ids, demo_ids = set(math["dbn"]), set(demo["dbn"])
        print(f"Math schools: {len(math_ids):,} | Demographics schools: {len(demo_ids):,}")
        print(f"Matched: {len(math_ids & demo_ids):,} | "
              f"Only in math: {len(math_ids - demo_ids):,} | "
              f"Only in demographics: {len(demo_ids - math_ids):,}")

        # Inner join: keep only schools that have both test results and demographics.
        # validate="one_to_one" stops the task if a DBN appears more than once on either side.
        df = math.merge(demo, on="dbn", how="inner", validate="one_to_one")

        # Data quality check: almost every math school should find its demographics
        match_rate = len(df) / len(math)
        if match_rate < 0.95:
            raise ValueError(f"Only {match_rate:.1%} of math schools matched; check the join key")

        # Record which school year the data describes, then put identifiers first
        df["school_year"] = "2021-22"
        first_columns = ["dbn", "school_year", "school_name"]
        df = df[first_columns + [c for c in df.columns if c not in first_columns]]

        print(f"Merged dataset: {len(df):,} schools, {len(df.columns)} columns "
              f"(match rate {match_rate:.1%})")
        print(df.head())

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "schools_merged.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------
    @task
    def load_to_postgres(merged_path: str) -> str:
        """Replace the contents of the warehouse table with the latest merged data."""
        import pandas as pd
        from airflow.providers.postgres.hooks.postgres import PostgresHook

        df = pd.read_csv(merged_path, dtype={"dbn": str})
        columns = ", ".join(df.columns)

        hook = PostgresHook(postgres_conn_id="schools_db")
        conn = hook.get_conn()
        try:
            with conn.cursor() as cur:
                # 1. Create the table on the first run (does nothing if it already exists)
                cur.execute(CREATE_TABLE_SQL)

                # 2. Remove the previous load so re-runs never create duplicate rows
                cur.execute(f"TRUNCATE TABLE {TABLE_NAME};")

                # 3. Bulk-load the CSV file with PostgreSQL's COPY command
                with open(merged_path) as f:
                    cur.copy_expert(
                        f"COPY {TABLE_NAME} ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true)",
                        f,
                    )

                # 4. Data quality check: the table must contain every merged row
                cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
                loaded = cur.fetchone()[0]
                if loaded != len(df):
                    raise ValueError(f"Loaded {loaded} rows, expected {len(df)}")

            conn.commit()      # Make all four steps permanent at once
        except Exception:
            conn.rollback()    # Undo everything, so the previous data stays intact
            raise
        finally:
            conn.close()

        print(f"Loaded {loaded:,} rows into table '{TABLE_NAME}'")
        return TABLE_NAME

    # -----------------------------------------------------------------
    # Analysis: visualization
    # -----------------------------------------------------------------
    @task
    def visualize(table_name: str) -> str:
        """Read the cleaned data from PostgreSQL and chart the main relationships."""
        import matplotlib
        matplotlib.use("Agg")                 # Draw to files: containers have no screen
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import PercentFormatter

        # 1. Read the data from the database (not from the CSV file)
        df = read_table(table_name)
        print(f"Read {len(df):,} rows from table '{table_name}'")

        # 2. Correlation of each demographic feature with the math proficiency rate
        features = ["economic_need_index", "pct_poverty", "pct_ell",
                    "pct_students_with_disabilities", "pct_female", "total_enrollment",
                    "pct_asian", "pct_black", "pct_hispanic", "pct_white", "pct_multi_racial"]
        correlations = df[features].corrwith(df["pct_proficient"]).sort_values()
        print("Correlation with pct_proficient (-1 to +1):")
        print(correlations.round(3).to_string())

        # 3. Derive the borough from the DBN and compare average proficiency
        df["borough"] = df["dbn"].str[2].map(BOROUGHS)
        by_borough = df.groupby("borough")["pct_proficient"].mean().sort_values()
        print("Average math proficiency by borough:")
        print(by_borough.round(3).to_string())

        # 4. Two-panel figure
        blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#fcfcfb")

        # Left: each dot is a school; the dark line is a linear trend
        x, y = df["economic_need_index"], df["pct_proficient"]
        r = x.corr(y)
        ax1.scatter(x, y, s=24, color=blue, alpha=0.45, edgecolors="none")
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax1.plot(xs, slope * xs + intercept, color=ink, linewidth=2)
        ax1.set_title(f"Higher economic need, lower math proficiency (r = {r:.2f})",
                      loc="left", color=ink, fontsize=12)
        ax1.set_xlabel("Economic Need Index", color=muted)
        ax1.set_ylabel("Students proficient in math (Level 3+4)", color=muted)

        # Right: average proficiency per borough, with direct value labels
        ax2.barh(by_borough.index, by_borough.values, color=blue, height=0.6)
        for i, value in enumerate(by_borough.values):
            ax2.text(value + 0.005, i, f"{value:.0%}", va="center", color=ink, fontsize=10)
        ax2.set_title("Average math proficiency by borough", loc="left", color=ink, fontsize=12)
        ax2.set_xlabel("Average share of students proficient", color=muted)

        # Shared styling: percent axes, light grid, no top/right borders
        ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        for ax in (ax1, ax2):
            ax.set_facecolor("#fcfcfb")
            ax.grid(color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=muted)

        fig.suptitle(f"NYC public schools, 2021-22 (n = {len(df):,})",
                     x=0.01, ha="left", color=muted, fontsize=10)
        fig.tight_layout()

        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ANALYSIS_DIR / "proficiency_analysis.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved chart -> {out_path}")
        return str(out_path)

    # -----------------------------------------------------------------
    # Analysis: baseline machine learning model
    # -----------------------------------------------------------------
    @task
    def train_model(table_name: str) -> str:
        """Train a baseline classifier that predicts above-median math proficiency."""
        import json

        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                     precision_score, recall_score, roc_auc_score)
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # 1. Read the data from the database
        df = read_table(table_name)
        print(f"Read {len(df):,} rows from table '{table_name}'")

        # 2. Target: 1 if the school's proficiency rate is above the median, else 0
        threshold = df["pct_proficient"].median()
        y = (df["pct_proficient"] > threshold).astype(int)
        X = df[MODEL_FEATURES]
        print(f"Target: pct_proficient > {threshold:.3f} (median)")
        print(f"Class balance: {y.value_counts().to_dict()}")

        # 3. Hold out 20% of schools as an untouched test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 4. Model: put features on the same scale, then logistic regression
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

        # 5. 5-fold cross-validation on the training set gives a more stable estimate
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        # 6. Train on the full training set and evaluate on the test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 7. Reference point: always predict the most common class
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

        # Coefficients on standardized features: sign = direction, size = strength
        coefficients = {
            name: round(float(c), 3)
            for name, c in zip(MODEL_FEATURES, model[-1].coef_[0])
        }

        metrics = {
            "model": "LogisticRegression (StandardScaler)",
            "features": MODEL_FEATURES,
            "target": f"pct_proficient > median ({threshold:.3f})",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 3),
            "cv_accuracy_std": round(float(cv_scores.std()), 3),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
            "test_precision": round(float(precision_score(y_test, y_pred)), 3),
            "test_recall": round(float(recall_score(y_test, y_pred)), 3),
            "test_f1": round(float(f1_score(y_test, y_pred)), 3),
            "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 3),
            "baseline_accuracy": round(float(baseline.score(X_test, y_test)), 3),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "coefficients": coefficients,
        }

        print(json.dumps(metrics, indent=2))

        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ANALYSIS_DIR / "model_metrics.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics -> {out_path}")
        return str(out_path)

    # -----------------------------------------------------------------
    # Task wiring
    # -----------------------------------------------------------------
    # Each dataset gets its own TaskGroup: extract -> clean -> transform.
    # The two groups do not depend on each other, so Airflow runs them in parallel.
    with TaskGroup(group_id="math", tooltip="Math test results: extract, clean, transform"):
        math_raw = extract.override(task_id="extract")("math")
        math_clean = clean_math.override(task_id="clean")(math_raw)
        math_ready = transform_math.override(task_id="transform")(math_clean)

    with TaskGroup(group_id="demographics", tooltip="School demographics: extract, clean, transform"):
        demographics_raw = extract.override(task_id="extract")("demographics")
        demographics_clean = clean_demographics.override(task_id="clean")(demographics_raw)
        demographics_ready = transform_demographics.override(task_id="transform")(demographics_clean)

    # Merge waits for BOTH groups because it needs both outputs
    merged = merge_datasets(math_ready, demographics_ready)

    # Load the merged result into PostgreSQL
    table = load_to_postgres(merged)

    # Both analysis tasks read from the database and do not depend on each other,
    # so they run in parallel after the load finishes
    chart = visualize(table)
    metrics = train_model(table)


nyc_schools_pipeline()