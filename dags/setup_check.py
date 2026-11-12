"""
Setup check DAG.

Verifies three things before building the main pipeline:
  1. Airflow can find and run DAG files from the dags/ folder.
  2. LocalExecutor runs independent tasks in PARALLEL.
  3. Airflow can connect to the schools_db data warehouse.
"""
from datetime import datetime
import time

from airflow.sdk import dag, task


@dag(
    dag_id="setup_check",
    schedule=None,                  # Manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,                  # Do not backfill past dates
    tags=["setup"],
)
def setup_check():

    @task
    def slow_task_a():
        # Sleeps for 10 seconds. If parallelism works,
        # this runs at the same time as slow_task_b.
        print("Task A started:", datetime.now())
        time.sleep(10)
        print("Task A finished:", datetime.now())

    @task
    def slow_task_b():
        print("Task B started:", datetime.now())
        time.sleep(10)
        print("Task B finished:", datetime.now())

    @task
    def check_schools_db():
        # Connect using the Airflow Connection defined in docker-compose.yaml
        from airflow.providers.postgres.hooks.postgres import PostgresHook

        hook = PostgresHook(postgres_conn_id="schools_db")
        db_name, db_version = hook.get_first("SELECT current_database(), version();")
        print(f"Connected to database '{db_name}': {db_version}")

    # A and B have no dependency on each other -> they run in parallel.
    # The DB check runs only after both have finished.
    [slow_task_a(), slow_task_b()] >> check_schools_db()


# Calling the function registers the DAG with Airflow
setup_check()