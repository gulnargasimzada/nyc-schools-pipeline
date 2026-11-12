# NYC Schools: Airflow Data Pipeline and Agent-Based ML Workflow

An end-to-end data pipeline built with **Apache Airflow** that ingests two related NYC Open Data datasets, cleans and merges them, loads the result into **PostgreSQL**, and analyzes it with a visualization and a baseline machine learning model. The cleaned data is then used in an agent-based ML workflow (Part 2).

**Question:** How is the economic profile of a NYC public school's student body related to its students' math results?

## Repository structure

```
nyc-schools-pipeline/
├── .devcontainer/
│   └── devcontainer.json        # VS Code Dev Container
├── dags/
│   ├── nyc_schools_pipeline.py  # Main pipeline DAG
│   └── setup_check.py           # Setup test DAG
├── agents/                      # Part 2: agent workflow
├── docker-compose.yaml          # Airflow + PostgreSQL services
├── Dockerfile                   # Custom Airflow image
├── requirements.txt             # Extra Python packages
├── .env                         # Airflow user ID
├── .gitignore                   # Files not stored in Git
└── README.md
```

The `data/` folder (downloads, intermediate files, charts, model metrics) is created by the pipeline at runtime and is not stored in the repository.

---

## Part 1: Airflow Data Pipeline

### Data sources

| Dataset | Source | Size (raw) | What it contains |
|---|---|---|---|
| [Math Test Results 2013-2023](https://data.cityofnewyork.us/d/74kb-55u9) | NYC Department of Education, NYC Open Data | 624,572 rows | NY State math exam results (grades 3-8) by school, grade, year and student group |
| [2017-18 to 2021-22 Demographic Snapshot](https://data.cityofnewyork.us/d/c7ru-d68s) | NYC Department of Education, NYC Open Data | 9,251 rows | Enrollment, English Language Learners, students with disabilities, poverty, Economic Need Index by school and year |

### Why these datasets

- **Meaningful question:** they connect learning outcomes with the economic and social context of each school, a central topic in education equity.
- **Related by a shared key:** both identify schools by their **DBN** (District-Borough-Number code, e.g. `01M015`), so they can be joined reliably without matching on school names.
- **Recent and aligned:** both cover the **2021-22 school year** (the spring 2022 math exam matches the 2021-22 demographic snapshot), the most recent year available in both.
- **Automatable:** both are public CSV downloads with no API key, so Airflow can fetch them on a schedule.
- **Realistic cleaning work:** mixed reporting levels, suppressed values and inconsistent formats made the cleaning and transformation steps meaningful.
- **Suitable for ML:** the proficiency rate gives a clear classification target, and the final dataset is small enough for an agent workflow to train and compare many models quickly.

### Pipeline structure



```
[math: extract -> clean -> transform] ----------+                                          +--> train_model
                                                +--> merge_datasets --> load_to_postgres --+
[demographics: extract -> clean -> transform] --+                                          +--> visualize
```

| Stage | Tasks | What happens |
|---|---|---|
| Ingest | `math.extract`, `demographics.extract` | Download each CSV from NYC Open Data |
| Clean | `math.clean`, `demographics.clean` | Filter to one row per school for 2021-22, fix data types, handle suppressed and censored values |
| Transform | `math.transform`, `demographics.transform` | Standardize column names and units, keep only the needed columns |
| Merge | `merge_datasets` | Join both datasets on `dbn` |
| Load | `load_to_postgres` | Write the result to the PostgreSQL table `school_math_demographics` |
| Analyze | `visualize`, `train_model` | Read from PostgreSQL, create charts, train and evaluate a model |

Each dataset's steps are grouped in a **TaskGroup**. Between tasks, only file paths and table names are passed through XCom; the data itself moves through files (`data/raw/` -> `data/staging/` -> `data/processed/`).

### Transformations

**Math results: 624,572 rows -> 1,078 schools**
- Kept school-level rows for the 2022 exam, all grades combined and all students (1,121 schools).
- Converted scores from text to numbers and dropped 43 schools whose results were suppressed (`s`).
- Renamed columns to snake_case (`Geographic Subdivision` -> `dbn`, `Pct Level 3 and 4` -> `pct_proficient`) and converted percentages to a 0-1 scale.

**Demographics: 9,251 rows -> 1,881 schools**
- Kept the 2021-22 school year and 13 of 44 columns.
- Converted text percentages (`"84.7%"`) to numbers and replaced censored `"Above 95%"` values with 0.95.
- Put all percentages on the same 0-1 scale.

**Merge and load:** inner join on `dbn` -> **1,078 schools, 17 columns, 100% of math schools matched**. Loaded into PostgreSQL with `dbn` as the primary key.

### Challenges and how they were solved

| Challenge | Solution |
|---|---|
| The math dataset has **no `DBN` column** | Exploration showed the DBN is stored in `Geographic Subdivision` for school rows; the column was renamed to `dbn` and the match was verified (100% of schools matched) |
| One table mixes citywide, borough, district and school rows, 7 grade levels and 30+ student groups | Filtered to `School`, `2022`, `All Grades`, `All Students`: one row per school |
| The two datasets write the school year differently (`2022` vs `2021-22`) | Mapped the spring 2022 exam to the 2021-22 school year |
| Suppressed results (`s`) in 43 schools | Dropped them, because the proficiency rate is the ML target and should not be imputed |
| Percentages in three formats (`0.042`, `"84.7%"`, `"Above 95%"`) | Converted all to 0-1 fractions; replaced censored `Above 95%` values (227 in total) with their lower bound instead of dropping the highest-need schools |
| Charter schools were planned as a feature, but none are reported at school level for 2022 | Removed the feature and documented the finding |
| The NYC Open Data server returned temporary `HTTP 503` errors on the 61 MB download | Added automatic retries (3 attempts, 1-minute delay) and safe downloads to a temporary file; during exploration, data was downloaded once and read from disk |
| Daily re-runs could duplicate rows in the database | `TRUNCATE` + `COPY` in a single transaction: re-runs always leave exactly 1,078 rows, and a failed load rolls back |
| Bad data could pass silently between steps | Data quality checks stop the run on duplicates, out-of-range percentages, a low merge match rate or a row-count mismatch |

### Scheduling and parallel processing

**Scheduling:** the DAG runs automatically every day at midnight UTC (`@daily`). `catchup=False` runs only the latest missed run after downtime, and `max_active_runs=1` prevents two runs from writing the same files at once.



**Parallel processing:** Airflow uses the **LocalExecutor**, which runs independent tasks at the same time.
- The `math` and `demographics` TaskGroups run in parallel: both downloads start in the same second, and the demographics group finishes completely while the larger math file is still downloading.
- After loading, `visualize` and `train_model` run in parallel.
- `merge_datasets` waits for both groups, because it needs both outputs.



### Results

Both analysis tasks read the cleaned data **from the PostgreSQL table**.

#### Visualization



- **Strong negative relationship (r = -0.78)** between a school's Economic Need Index and the share of its students proficient in math.
- **Borough averages:** Queens 45%, Staten Island 44%, Manhattan 43%, Brooklyn 35%, **Bronx 21%**.
- Other correlations with proficiency: poverty -0.76, students with disabilities -0.49, English Language Learners -0.20, share of female students 0.05 (no relationship).

#### Baseline model

Logistic Regression predicting whether a school's math proficiency is **above the median (30.8%)**, using five features: Economic Need Index, % English Language Learners, % students with disabilities, % female, total enrollment. Poverty (duplicates the Economic Need Index), race/ethnicity shares (proxies for economic need, fairness concerns) and test-derived columns (data leakage) were excluded on purpose.

| Metric | Value |
|---|---|
| **Test accuracy** | **0.810** |
| Cross-validation accuracy (5-fold) | 0.835 ± 0.018 |
| ROC-AUC | 0.888 |
| Precision / Recall / F1 | 0.813 / 0.806 / 0.809 |
| Majority-class baseline | 0.500 |



#### Key findings

- From five demographic indicators alone, the model predicts a school's above/below-median math result with **81% accuracy**, 31 points above the baseline. The **Economic Need Index** is by far the strongest predictor (standardized coefficient -2.53).
- The Bronx, the borough with the highest economic need, has the lowest average proficiency.
- The model's errors point to schools that do **better (21) or worse (20) than their demographics predict**, which are the most interesting cases for further study.
- These are **school-level associations**: they do not describe individual students and do not show that economic need causes lower results. Factors not in the data, such as school resources and teacher experience, may explain part of the gap.

**Limitations:** one school year; school-level data only; the median threshold is a modeling choice; censored values were set to 95%; charter schools are not included.

---

## Part 2: Agent-Based ML Workflow

_To be completed._

---

## How to run

### Prerequisites
- Docker Desktop (at least 4 GB of memory for containers)
- Free ports `8080` (Airflow UI) and `5433` (schools database)

### Part 1: Airflow pipeline

```bash
git clone <repository-url>
cd nyc-schools-pipeline

docker compose build              # Build the Airflow image
docker compose up airflow-init    # Initialize Airflow and create the admin user
docker compose up -d              # Start all services
```

1. Open http://localhost:8080 and log in with `airflow` / `airflow`.
2. Unpause `nyc_schools_pipeline`. It runs daily; click **Trigger** to run it now (about 2-5 minutes).
3. Outputs:
   - PostgreSQL table: `docker compose exec schools-db psql -U schools -d schools -c "SELECT COUNT(*) FROM school_math_demographics;"`
   - Chart: `data/analysis/proficiency_analysis.png`
   - Model metrics: `data/analysis/model_metrics.json`

To stop: `docker compose stop` (keeps data). `docker compose down -v` also deletes the databases.

On Linux, set your user ID before the first start: `echo "AIRFLOW_UID=$(id -u)" > .env`.

**Dev Container (optional):** open the folder in VS Code and run **Dev Containers: Reopen in Container** to work inside the Airflow environment.

### Part 2: Agent workflow
_To be completed._

---